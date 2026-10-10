"""ds005170 (Chisco) as a :class:`~bci_eeg_to_text.sources.DatasetSource`.

Chisco is **output-only**: the marker channel carries one uniform reward pulse
per imagined sentence and no stimulus-onset triggers. The pairing between EEG
and text is therefore *recovered* rather than read:

    run ``run-NNN``  <->  ``textdataset/split_data_N.xlsx``
    i-th reward pulse <-> i-th sentence in that file

See issue #1 for the evidence and issue #4 for why this is now one of two
sources behind a common interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bci_eeg_to_text.data import write_trial_cache
from bci_eeg_to_text.eeg import (
    NON_EEG_CHANNELS,
    TRIGGER_CHANNEL,
    AlignmentError,
    RunId,
    align_segments,
    find_pulses,
    is_annex_pointer,
    read_metadata,
    read_signals,
)
from bci_eeg_to_text.sources import (
    ExtractionReport,
    RunInfo,
    load_participants_tsv,
    register_source,
    usable_subjects,
)
from bci_eeg_to_text.splits import Participant
from bci_eeg_to_text.textdata import read_split, split_file_for_run

#: Filenames look like ``sub-01_ses-02_task-imagine_run-010_eeg.edf``.
_GLOB = "sub-*/ses-*/eeg/*_eeg.edf"


@register_source
@dataclass
class ChiscoSource:
    """Aligned trials from ds005170, recovered from reward pulses."""

    name = "chisco"
    #: Default location; the dataset is read-only and never modified.
    data_root: Path = Path("eeg-text-bci-data/bci-eeg/ds005170")
    #: Samples kept before each pulse. The pulse is assumed to mark trial onset.
    pre_samples: int = 0
    post_samples: int | None = None
    decimate: int = 8

    def root(self) -> Path:
        return Path(self.data_root)

    def _all_paths(self) -> list[Path]:
        return sorted(self.root().glob(_GLOB))

    def runs(self) -> list[RunInfo]:
        """Usable runs only; placeholders are skipped silently.

        The corpus is partially downloaded by design (the drive is small), so an
        unfetched file is an expected condition, not an error.
        """
        out: list[RunInfo] = []
        for path in self._all_paths():
            if is_annex_pointer(path):
                continue
            run_id = RunId.parse(path.name)
            out.append(
                RunInfo(
                    subject_id=run_id.subject,
                    dataset=self.name,
                    run=f"run-{run_id.run:03d}",
                    path=path,
                    session=f"ses-{run_id.session:02d}",
                )
            )
        return out

    def participants(self) -> list[Participant]:
        declared = load_participants_tsv(self.root() / "participants.tsv")
        return usable_subjects(declared, self.runs())

    def extract(self, run: RunInfo, out_path: Path, **options: object) -> ExtractionReport:
        decimate = int(options.get("decimate", self.decimate))
        pre = int(options.get("pre_samples", self.pre_samples))
        post = options.get("post_samples", self.post_samples)

        # Read the single Trigger channel first: cheap, and it lets misalignment
        # fail before paying for the full 128-channel read.
        trigger, sample_rate = read_signals(run.path, [TRIGGER_CHANNEL])
        pulses = find_pulses(trigger[0])

        run_id = RunId.parse(run.path.name)
        texts = read_split(split_file_for_run(self.root() / "textdataset", run_id.run))
        if len(pulses) != len(texts):
            raise AlignmentError(
                f"{run_id.key}: reward pulses ({len(pulses)}) != stimulus sentences "
                f"({len(texts)}); refusing to produce misaligned pairs (issue #1)"
            )

        meta = read_metadata(run.path)
        eeg_names = [c for c in meta.channel_names if c not in NON_EEG_CHANNELS]
        signals, sample_rate = read_signals(run.path, eeg_names)

        segments = align_segments(
            pulses, texts, n_samples=signals.shape[1], pre_samples=pre, post_samples=post
        )
        trials = [signals[:, s.start_sample : s.stop_sample] for s in segments]
        if decimate > 1:
            trials = [t[:, ::decimate] for t in trials]
            sample_rate = sample_rate / decimate

        write_trial_cache(
            out_path,
            trials,
            [s.text for s in segments],
            [s.n_samples for s in segments],
            [s.start_sample for s in segments],
            {
                "sample_rate": np.array(sample_rate),
                "subject": np.array(run.subject_id),
                "run": np.array(run_id.run),
                "task": np.array(run_id.task),
                "dataset": np.array(self.name),
                "alignment": np.array("reward-pulse"),
            },
        )
        return ExtractionReport(
            run=run,
            n_segments=len(segments),
            details={
                "channels": trials[0].shape[0],
                "sample_rate": sample_rate,
                "alignment": "reward-pulse",
            },
        )

    def describe(self) -> str:
        n_runs = len(self.runs())
        n_subjects = len({r.subject_id for r in self.runs()})
        return (
            f"ds005170 Chisco (imagined speech, alignment recovered from reward "
            f"pulses): {n_subjects} usable subjects, {n_runs} usable runs"
        )
