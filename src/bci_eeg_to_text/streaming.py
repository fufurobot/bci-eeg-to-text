"""Streaming access to `(EEG trial, text row)` pairs, without copying data.

The corpora are 419 GB across two OpenNeuro datasets on a read-only mount.
Extracting trials into a derived cache duplicates tens of gigabytes of the same
bytes, and on a machine with ~15 GB free that is not a rounding error -- it is
the difference between being able to run at all and not.

So nothing is copied. Instead:

1. **Index** is built from cheap metadata only: an events file parses in ~0.03 s,
   so every run's row intervals and texts are enumerated without touching EEG
   bytes. This costs kilobytes and is rebuilt on demand.
2. **Reads** are random-access slices straight out of the original recording:
   opening a BrainVision file takes ~0.5 s and a 4 s trial slice ~0.14 s, so a
   sample costs roughly a tenth of a second on a cold cache and far less when
   the OS page cache is warm.

The result is an ordinary `torch.utils.data.Dataset` that handlers, samplers and
accelerate can treat as if the data were small.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from bci_eeg_to_text.chineseeeg import (
    read_events,
    read_segmented_novel,
    readable_vhdr,
    row_intervals,
)
from bci_eeg_to_text.eeg import is_annex_pointer
from bci_eeg_to_text.tokenizer import CharTokenizer


@dataclass(frozen=True)
class EchoTrial:
    """Where one supervised trial lives, without holding its samples."""

    run_path: Path
    dataset: str
    subject: str
    run: str
    session: str | None
    start: int
    stop: int
    text: str
    row_index: int

    @property
    def key(self) -> str:
        return f"{self.dataset}:{self.subject}/{self.session}/{self.run}#{self.row_index}"

    @property
    def n_samples(self) -> int:
        return self.stop - self.start


def _novel_for(run_path: Path) -> tuple[Path | None, str | None]:
    """Find the text workbook and source novel a run belongs to."""
    parts = run_path.parts
    try:
        derivs = parts.index("derivatives")
    except ValueError:
        return None, None
    root = Path(*parts[: derivs + 1])
    name = run_path.name
    novel = "LittlePrince" if "LittlePrince" in name else "GarnettDream"
    run_no = None
    for token in name.replace(".", "_").split("_"):
        if token.startswith("run-") and token[4:].isdigit():
            run_no = int(token[4:])
            break
    if run_no is None:
        return None, None
    seg_dir = root / "novels" / "segmented_novel" / novel
    if not seg_dir.exists():
        return None, None
    import re

    pattern = re.compile(rf"run_{run_no}(?!\d)")
    for path in sorted(seg_dir.glob("*.xlsx")):
        if "_display" in path.name:
            continue
        if pattern.search(path.stem):
            return path, novel
    return None, None


def index_chineseeeg(
    data_root: Path,
    *,
    subjects: Sequence[str] | None = None,
    sessions: Sequence[str] | None = None,
    verbose: bool = True,
) -> list[EchoTrial]:
    """Enumerate every supervised row of a ChineseEEG tree, reading metadata only.

    Row intervals come from ``_events.tsv`` and texts from the segmented novel
    workbook; neither involves the EEG itself, so indexing the whole corpus is
    cheap and touches no large file.
    """
    from bci_eeg_to_text.sources import get_source

    source = get_source("chineseeeg", data_root=data_root)
    trials: list[EchoTrial] = []
    # Reading a workbook costs seconds (openpyxl on a 250 KB xlsx measured ~9 s
    # here, and importing pandas ~28 s), so each file is parsed once and reused
    # across the runs that share it.
    novel_cache: dict[Path, list[str]] = {}
    # Keyed by (subject, text file): every participant reads the SAME novel, so
    # deduplicating by text file alone would index only the first subject.
    seen: set[tuple[str, Path]] = set()

    for run in source.runs():
        if subjects and run.subject_id not in subjects:
            continue
        if sessions and run.session not in sessions:
            continue
        text_path, _novel = _novel_for(run.path)
        if text_path is None:
            continue
        dedup_key = (run.subject_id, text_path)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        # Unfetched git-annex stubs look like real files but hold only a pointer
        # line. BOTH the header and the .eeg payload must be checked: a real
        # .vhdr next to a stubbed .eeg parses without error and then reports
        # zero samples, which surfaces much later as "No data in this range".
        if is_annex_pointer(run.path) or is_annex_pointer(run.path.with_suffix(".eeg")):
            if verbose:
                print(f"  skip {run.subject_id} {run.run}: annex pointer (not fetched)", flush=True)
            continue

        events_path = source.events_path(run)
        if not events_path.exists():
            continue
        try:
            intervals = row_intervals(read_events(events_path))
            texts = novel_cache.get(text_path)
            if texts is None:
                texts = read_segmented_novel(text_path)
                novel_cache[text_path] = texts
        except Exception:  # noqa: BLE001 - a bad run must not stop indexing
            continue
        if len(intervals) != len(texts):
            if verbose:
                print(
                    f"  skip {run.subject_id} {run.session} {run.run}: "
                    f"{len(intervals)} intervals vs {len(texts)} rows",
                    flush=True,
                )
            continue
        for i, ((start, stop), text) in enumerate(zip(intervals, texts, strict=True)):
            trials.append(
                EchoTrial(
                    run_path=run.path,
                    dataset="chineseeeg",
                    subject=run.subject_id,
                    run=run.run,
                    session=run.session,
                    start=start,
                    stop=stop,
                    text=text,
                    row_index=i,
                )
            )
    return trials


class RawRunCache:
    """Keeps a few opened recordings in memory, as raw file handles.

    Opening a BrainVision header takes ~0.5 s. With shuffled sampling that would
    dominate, so the most recently used recordings are kept open (not preloaded:
    no samples are held, only the file handle and index).
    """

    def __init__(self, capacity: int = 8, scratch: Path | None = None) -> None:
        self.capacity = capacity
        self.scratch = scratch or Path(".cache/vhdr_repair")
        self._open: dict[Path, object] = {}
        self._order: list[Path] = []

    def get(self, path: Path):
        cached = self._open.get(path)
        if cached is not None:
            self._order.remove(path)
            self._order.append(path)
            return cached

        import mne

        vhdr, _ = readable_vhdr(path, self.scratch)
        raw = mne.io.read_raw_brainvision(str(vhdr), preload=False, verbose="ERROR")
        self._open[path] = raw
        self._order.append(path)
        while len(self._order) > self.capacity:
            evicted = self._order.pop(0)
            self._open.pop(evicted, None)
        return raw

    def close(self) -> None:
        self._open.clear()
        self._order.clear()


class StreamingTrialDataset(Dataset):
    """Yields fixed-length EEG windows read straight from the originals.

    ``n_samples`` crops or zero-pads along time so trials batch together; the
    true pre-padding length is returned so CTC never reads padding as signal.
    """

    def __init__(
        self,
        trials: Sequence[EchoTrial],
        tokenizer: CharTokenizer,
        *,
        n_samples: int = 1152,
        decimate: int = 1,
        cache: RawRunCache | None = None,
        max_target_length: int = 64,
    ) -> None:
        if not trials:
            raise ValueError("no trials to stream")
        self.trials = list(trials)
        self.tokenizer = tokenizer
        self.n_samples = n_samples
        self.decimate = decimate
        self.cache = cache or RawRunCache()
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.trials)

    def __getitem__(self, index: int) -> dict[str, object]:
        trial = self.trials[index]
        raw = self.cache.get(trial.run_path)
        # Row intervals come from the event file, and the final rows can run past
        # the recording when its annotations disagree with its sample count by a
        # few frames. Clamp rather than letting mne raise "No data in this range".
        start = max(0, min(int(trial.start), raw.n_times))
        stop = max(start + 1, min(int(trial.stop), raw.n_times))
        data = raw.get_data(start=start, stop=stop)
        if self.decimate > 1:
            data = data[:, :: self.decimate]

        n_channels, length = data.shape
        window = np.zeros((n_channels, self.n_samples), dtype=np.float32)
        keep = min(length, self.n_samples)
        window[:, :keep] = data[:, :keep]

        target = self.tokenizer.encode(trial.text)[: self.max_target_length]
        if not target:
            # CTC is undefined for an empty target; substitute the blank token's
            # absence by dropping the row rather than crashing mid-epoch.
            target = [self.tokenizer.unk]
        return {
            "eeg": torch.from_numpy(window),
            "target": torch.tensor(target, dtype=torch.long),
            "target_length": len(target),
            "text": trial.text,
            "key": trial.key,
        }


def collate(batch: Sequence[dict[str, object]]) -> dict[str, object]:
    """Pad targets and record lengths for ``CTCLoss``."""
    eeg = torch.stack([b["eeg"] for b in batch])
    targets = [b["target"] for b in batch]
    max_len = max(int(t.shape[0]) for t in targets)
    padded = torch.zeros(len(targets), max_len, dtype=torch.long)
    for i, t in enumerate(targets):
        padded[i, : t.shape[0]] = t
    return {
        "eeg": eeg,
        "targets": padded,
        "target_lengths": torch.tensor([b["target_length"] for b in batch]),
        "input_lengths": torch.tensor([eeg.shape[-1]] * len(batch)),
        "texts": [b["text"] for b in batch],
    }


def split_by_subject(
    trials: Sequence[EchoTrial],
    *,
    valid_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[list[EchoTrial], list[EchoTrial]]:
    """Hold out whole participants, never trials from a trained-on one.

    Splitting trials at random would let a model see other rows of the same
    recording (indeed of the same paragraph) during training, which is the
    leakage the leakage matrix exists to catch.
    """
    import random

    subjects = sorted({t.subject for t in trials})
    if len(subjects) < 2:
        # Single subject: hold out whole runs instead.
        runs = sorted({(t.subject, t.session, t.run) for t in trials})
        random.Random(seed).shuffle(runs)
        n_valid = max(1, int(len(runs) * valid_fraction))
        held = set(runs[:n_valid])
        train = [t for t in trials if (t.subject, t.session, t.run) not in held]
        valid = [t for t in trials if (t.subject, t.session, t.run) in held]
        return train, valid

    shuffled = subjects[:]
    random.Random(seed).shuffle(shuffled)
    n_valid = max(1, int(round(len(shuffled) * valid_fraction)))
    held = set(shuffled[:n_valid])
    train = [t for t in trials if t.subject not in held]
    valid = [t for t in trials if t.subject in held]
    return train, valid


def text_fingerprint(texts: Iterator[str]) -> str:
    """Stable hash of a text corpus, for recording what a run actually saw."""
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:16]
