"""ds004952 (ChineseEEG) as a :class:`~bci_eeg_to_text.sources.DatasetSource`.

ChineseEEG is a **silently-read** corpus: participants read two Chinese novels
while 128-channel EEG was recorded at 256 Hz. Unlike ds005170 it ships real
stimulus markers, so the EEG-to-text pairing is *read* from the events rather
than inferred.

Marker semantics (from the dataset README and verified against the data):

===========  ==================================================
``ROWS`` /   begin / end of a displayed row of text
``ROWE``
``CH01..``   a chapter begins
``bad``      a segment marked bad by the dataset authors
===========  ==================================================

Verified timing property: a row's duration equals its number of highlighted
characters times 0.35 s (11 chars -> 3.96 s, 6 chars -> 2.02 s). Rows contain
at most ten characters, and the durations observed are multiples of ~0.35 s.

.. warning::

   The naive assumption "the i-th ``ROWS`` event corresponds to the i-th text
   line" is **not** safe for this dataset. For ``sub-04`` run 1 the events
   contain 397 ``ROWS`` while the accompanying text workbook has 391 lines, and
   summed row durations imply ~4103 character slots against 3956 characters in
   the workbook. The mismatch is systematic rather than random, and this module
   therefore does **not** zip the two together blindly: it validates first and
   raises on disagreement. See issue #4.
"""

from __future__ import annotations

import csv
import re
import zipfile
from dataclasses import dataclass
from html import unescape
from pathlib import Path

import numpy as np

from bci_eeg_to_text.data import write_trial_cache
from bci_eeg_to_text.eeg import AlignmentError, Segment
from bci_eeg_to_text.sources import (
    ExtractionReport,
    RunInfo,
    load_participants_tsv,
    register_source,
    usable_subjects,
)
from bci_eeg_to_text.splits import Participant
from bci_eeg_to_text.tokenizer import normalize_text

#: Seconds each character is highlighted during presentation.
CHARACTER_DURATION_S = 0.35

#: Preprocessed variants shipped by the dataset, most-filtered last.
PREPROC_VARIANTS = ("filtered_0.5_80", "filtered_0.5_30")

_TEXT_ROW_RE = re.compile(r"<row[^>]*>(.*?)</row>", re.S)
_INLINE_RE = re.compile(r"<is><t[^>]*>(.*?)</t></is>", re.S)


@dataclass(frozen=True)
class Event:
    """One row of a BIDS ``_events.tsv``."""

    onset_s: float
    duration_s: float
    trial_type: str
    sample: int


def read_events(path: str | Path) -> list[Event]:
    """Read a BIDS events file.

    Opened with ``utf-8-sig`` because these files carry a BOM which otherwise
    corrupts the first column name.
    """
    events: list[Event] = []
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            try:
                events.append(
                    Event(
                        onset_s=float(row["onset"]),
                        duration_s=float(row.get("duration") or 0.0),
                        trial_type=(row.get("trial_type") or "").strip(),
                        sample=int(float(row.get("sample") or 0)),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise AlignmentError(f"{Path(path).name}: malformed event row {row!r}") from exc
    return events


def row_intervals(events: list[Event]) -> list[tuple[int, int]]:
    """Turn ``ROWS``/``ROWE`` markers into ``(start_sample, stop_sample)`` pairs.

    A ``ROWS`` without a following ``ROWE`` is an error: silently dropping it
    would shift every later row's text by one.
    """
    starts = [e.sample for e in events if e.trial_type == "ROWS"]
    stops = [e.sample for e in events if e.trial_type == "ROWE"]
    if not starts:
        return []

    intervals: list[tuple[int, int]] = []
    stop_cursor = 0
    for start in starts:
        while stop_cursor < len(stops) and stops[stop_cursor] <= start:
            stop_cursor += 1
        if stop_cursor >= len(stops):
            raise AlignmentError(
                f"ROWS at sample {start} has no following ROWE; the event stream is "
                "unbalanced and row-to-text alignment cannot be trusted"
            )
        intervals.append((start, stops[stop_cursor]))
        stop_cursor += 1
    return intervals


def read_segmented_novel(path: str | Path) -> list[str]:
    """Read the displayed text lines of a ``segmented_*_run_N.xlsx`` workbook.

    These workbooks hold a single column: a header, numeric cells marking
    chapter positions, and one cell per displayed text line.
    """
    with zipfile.ZipFile(path) as archive:
        sheet = archive.read("xl/worksheets/sheet1.xml").decode("utf-8")

    texts: list[str] = []
    for row_xml in _TEXT_ROW_RE.findall(sheet):
        cells = [unescape(c) for c in _INLINE_RE.findall(row_xml)]
        if not cells:
            continue
        cell = cells[0].strip()
        if not cell:
            continue
        # Numeric cells are chapter indices, not readable text.
        if cell.isdigit():
            continue
        texts.append(normalize_text(cell))
    return texts


@register_source
@dataclass
class ChineseEEGSource:
    """Aligned trials from ds004952 (silent reading)."""

    name = "chineseeeg"
    data_root: Path = Path("eeg-text-bci-data/bci-eeg/ds004952")
    #: Which preprocessed variant to read; falls back to the next if absent.
    variant: str = PREPROC_VARIANTS[0]
    #: Downsample factor applied to the 256 Hz signal when caching.
    decimate: int = 4

    def root(self) -> Path:
        return Path(self.data_root)

    def preproc_root(self) -> Path:
        """Directory holding the preprocessed variant actually present."""
        base = self.root() / "derivatives" / "preproc"
        for variant in (self.variant, *PREPROC_VARIANTS):
            candidate = base / variant
            if candidate.exists():
                return candidate
        return base / self.variant

    def _novel_dir(self) -> Path:
        return self.root() / "derivatives" / "novels" / "segmented_novel"

    def runs(self) -> list[RunInfo]:
        """Every preprocessed reading run, sorted deterministically."""
        out: list[RunInfo] = []
        for vhdr in sorted(self.preproc_root().glob("sub-*/ses-*/eeg/*_task-reading_*_eeg.vhdr")):
            match = re.match(
                r"(sub-\d+)_(ses-[A-Za-z]+)_task-reading_run-(\d+)_eeg\.vhdr", vhdr.name
            )
            if not match:
                continue
            subject, session, run = match.groups()
            out.append(
                RunInfo(
                    subject_id=subject,
                    dataset=self.name,
                    run=f"run-{int(run):02d}",
                    path=vhdr,
                    session=session,
                )
            )
        return out

    def participants(self) -> list[Participant]:
        declared = load_participants_tsv(self.root() / "participants.tsv")
        return usable_subjects(declared, self.runs())

    def events_path(self, run: RunInfo) -> Path:
        return run.path.with_name(run.path.name.replace("_eeg.vhdr", "_events.tsv"))

    def novel_paths(self, run: RunInfo) -> list[Path]:
        """Candidate text workbooks for a run.

        Sessions map to different novels (``ses-LittlePrince`` /
        ``ses-GarnettDream``), and the run number selects the chapter file.
        """
        novel = "LittlePrince" if "LittlePrince" in (run.session or "") else "GarnettDream"
        directory = self._novel_dir() / novel
        if not directory.exists():
            return []
        run_no = int(run.run.split("-")[1])
        return sorted(directory.glob(f"*run_{run_no}*.xlsx"))

    def extract(self, run: RunInfo, out_path: Path, **options: object) -> ExtractionReport:
        import mne

        decimate = int(options.get("decimate", self.decimate))
        events = read_events(self.events_path(run))
        intervals = row_intervals(events)
        if not intervals:
            return ExtractionReport(run=run, n_segments=0, skipped_reason="no ROWS/ROWE markers")

        texts = self._texts_for_run(run)
        if len(texts) != len(intervals):
            raise AlignmentError(
                f"{run.subject_id} {run.session} {run.run}: {len(intervals)} ROWS/ROWE "
                f"intervals but {len(texts)} text lines. Naive row-to-text zipping is "
                "NOT safe for this dataset; the mapping must be resolved first "
                "(see issue #4)."
            )

        raw = mne.io.read_raw_brainvision(str(run.path), preload=True, verbose="ERROR")
        data = raw.get_data()
        segments = [
            Segment(index=i, start_sample=start, stop_sample=stop, text=text)
            for i, ((start, stop), text) in enumerate(zip(intervals, texts, strict=True))
        ]

        # Keep trials and their text together: dropping an empty trial without
        # dropping its segment would silently desync every later pairing.
        kept_segments = [s for s in segments if s.stop_sample > s.start_sample]
        if not kept_segments:
            return ExtractionReport(run=run, n_segments=0, skipped_reason="empty segments")
        trials = [data[:, s.start_sample : s.stop_sample] for s in kept_segments]
        if decimate > 1:
            trials = [t[:, ::decimate] for t in trials]

        write_trial_cache(
            out_path,
            trials,
            [s.text for s in kept_segments],
            [s.n_samples for s in kept_segments],
            [s.start_sample for s in kept_segments],
            {
                "sample_rate": np.array(float(raw.info["sfreq"]) / decimate),
                "subject": np.array(run.subject_id),
                "run": np.array(int(run.run.split("-")[1])),
                "task": np.array("reading"),
                "dataset": np.array(self.name),
                "alignment": np.array("stimulus-marker"),
            },
        )
        return ExtractionReport(
            run=run,
            n_segments=len(trials),
            details={
                "channels": trials[0].shape[0],
                "sample_rate": float(raw.info["sfreq"]) / decimate,
                "alignment": "stimulus-marker",
            },
        )

    def _texts_for_run(self, run: RunInfo) -> list[str]:
        """Text lines for a run, preferring the plain workbook over ``_display``.

        The ``_display`` workbooks describe per-character presentation but are
        incomplete for this purpose (one run's file covers only the first two
        displayed rows), so the plain workbook is authoritative here.
        """
        candidates = [p for p in self.novel_paths(run) if "_display" not in p.name]
        if not candidates:
            candidates = self.novel_paths(run)
        if not candidates:
            return []
        return read_segmented_novel(candidates[0])

    def describe(self) -> str:
        runs = self.runs()
        subjects = sorted({r.subject_id for r in runs})
        return (
            f"ds004952 ChineseEEG (silent reading, stimulus markers): "
            f"{len(subjects)} subjects, {len(runs)} runs"
        )
