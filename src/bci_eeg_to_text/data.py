"""Building aligned ``(EEG trial, sentence)`` examples from the Chisco dataset.

The pipeline is deliberately split in two stages:

1. :func:`extract_run` reads one EDF run, recovers its reward pulses, pairs them
   with the stimulus sentences and writes a compact ``.npz``. Raw EDFs are
   hundreds of megabytes each and are stored on a read-only mount, so this is
   done once and cached.
2. :class:`TrialDataset` reads those cached arrays and yields fixed-length
   training examples.

Splitting this way keeps the expensive I/O out of the training loop and makes
the alignment auditable: the cache records which sentences were paired with
which pulses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bci_eeg_to_text.eeg import (
    NON_EEG_CHANNELS,
    TRIGGER_CHANNEL,
    AlignmentError,
    RunId,
    align_segments,
    find_pulses,
    read_metadata,
    read_signals,
)
from bci_eeg_to_text.textdata import read_split, split_file_for_run
from bci_eeg_to_text.tokenizer import CharTokenizer


@dataclass(frozen=True)
class ExtractConfig:
    """Settings for turning one run into cached trials."""

    #: Samples kept before the pulse onset. The pulse is assumed to mark the
    #: trial onset (see issue #1); a small lead-in preserves any pre-stimulus
    #: baseline.
    pre_samples: int = 0
    #: Samples kept after the onset. ``None`` uses the gap to the next pulse.
    post_samples: int | None = None
    #: Downsample factor applied while extracting, to cut cache size.
    decimate: int = 1


def _eeg_channel_names(path: Path) -> list[str]:
    meta = read_metadata(path)
    return [c for c in meta.channel_names if c not in NON_EEG_CHANNELS]


def extract_run(
    edf_path: str | Path,
    dataset_root: str | Path,
    out_path: str | Path,
    *,
    config: ExtractConfig | None = None,
) -> dict[str, object]:
    """Extract aligned trials from one run and cache them to ``out_path``.

    Raises :class:`AlignmentError` if the pulse count does not match the
    stimulus sentence count, rather than producing mislabelled training data.
    """
    config = config or ExtractConfig()
    edf_path = Path(edf_path)
    dataset_root = Path(dataset_root)
    out_path = Path(out_path)

    # Read the trigger channel first: it is a single channel, so this is fast
    # even when the mount is slow, and it lets alignment fail early before
    # paying for the expensive full-band read.
    trigger, sample_rate = read_signals(edf_path, [TRIGGER_CHANNEL])
    pulses = find_pulses(trigger[0])

    run_id = RunId.parse(edf_path.name)
    split_path = split_file_for_run(dataset_root / "textdataset", run_id.run)
    texts = read_split(split_path)
    if len(pulses) != len(texts):
        raise AlignmentError(
            f"{run_id.key}: reward pulses ({len(pulses)}) != target sentences "
            f"({len(texts)}); refusing to train on misaligned pairs - see issue #1"
        )

    meta = read_metadata(edf_path)
    eeg_names = [c for c in meta.channel_names if c not in NON_EEG_CHANNELS]
    signals, sample_rate = read_signals(edf_path, eeg_names)
    eeg = signals

    segments = align_segments(
        pulses,
        texts,
        n_samples=eeg.shape[1],
        pre_samples=config.pre_samples,
        post_samples=config.post_samples,
    )

    trials = [eeg[:, s.start_sample : s.stop_sample] for s in segments]
    if config.decimate > 1:
        trials = [t[:, :: config.decimate] for t in trials]
        sample_rate = sample_rate / config.decimate

    # Trials have different lengths, so they cannot be stacked into one dense
    # array. Store them concatenated along time with offsets, which keeps the
    # cache compact and lossless.
    flat = np.concatenate([t.astype(np.float32) for t in trials], axis=1)
    offsets = np.zeros(len(trials) + 1, dtype=np.int64)
    np.cumsum([t.shape[1] for t in trials], out=offsets[1:])
    n_channels = trials[0].shape[0]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        flat=flat,
        offsets=offsets,
        n_channels=np.array(n_channels),
        texts=np.array([s.text for s in segments], dtype=object),
        lengths=np.array([s.n_samples for s in segments], dtype=np.int64),
        onsets=pulses,
        sample_rate=np.array(sample_rate),
        subject=np.array(meta.run_id.subject),
        run=np.array(meta.run_id.run),
        task=np.array(meta.run_id.task),
    )
    return {
        "subject": meta.run_id.subject,
        "run": meta.run_id.run,
        "n_trials": len(segments),
        "shape": (len(segments), n_channels, int(offsets[-1] // max(1, len(segments)))),
        "path": str(out_path),
    }


@dataclass(frozen=True)
class Trial:
    """One training example: a fixed-length EEG window and its target ids."""

    eeg: np.ndarray
    target: np.ndarray
    text: str


class TrialDataset:
    """Fixed-length, padded view over cached run arrays.

    Trials are cropped or padded along time to a common ``n_samples`` so they
    can be batched. ``lengths`` records the true pre-padding length, which CTC
    needs to avoid reading padding as signal.
    """

    def __init__(
        self,
        cache_paths: list[str | Path],
        tokenizer: CharTokenizer,
        *,
        n_samples: int = 8000,
        max_target_length: int = 64,
    ) -> None:
        self.tokenizer = tokenizer
        self.n_samples = n_samples
        self.max_target_length = max_target_length
        self._index: list[tuple[Path, int]] = []
        for path in cache_paths:
            with np.load(path, allow_pickle=True) as data:
                for i in range(len(data["texts"])):
                    self._index.append((Path(path), i))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> Trial:
        path, row = self._index[i]
        with np.load(path, allow_pickle=True) as data:
            offsets = data["offsets"]
            flat = data["flat"]
            trial = flat[:, offsets[row] : offsets[row + 1]]
            text = str(data["texts"][row])

        eeg = _fit_length(trial, self.n_samples)
        ids = self.tokenizer.encode(text)[: self.max_target_length]
        if not ids:
            raise AlignmentError(f"example {i} has an empty target: {text!r}")
        return Trial(eeg=eeg, target=np.array(ids, dtype=np.int64), text=text)


def _fit_length(trial: np.ndarray, n_samples: int) -> np.ndarray:
    """Crop or zero-pad ``trial`` along time to exactly ``n_samples``."""
    n_channels, length = trial.shape
    if length == n_samples:
        return trial.astype(np.float32)
    if length > n_samples:
        return trial[:, :n_samples].astype(np.float32)
    out = np.zeros((n_channels, n_samples), dtype=np.float32)
    out[:, :length] = trial
    return out


def collate(batch: list[Trial]) -> dict[str, np.ndarray]:
    """Stack trials into padded arrays plus CTC length bookkeeping."""
    eeg = np.stack([t.eeg for t in batch])
    targets = np.concatenate([t.target for t in batch])
    target_lengths = np.array([len(t.target) for t in batch], dtype=np.int64)
    input_lengths = np.array([t.eeg.shape[1] for t in batch], dtype=np.int64)
    return {
        "eeg": eeg,
        "targets": targets,
        "target_lengths": target_lengths,
        "input_lengths": input_lengths,
        "texts": np.array([t.text for t in batch], dtype=object),
    }


def discover_runs(dataset_root: str | Path) -> list[Path]:
    """List every EDF run in the dataset, sorted for reproducibility."""
    root = Path(dataset_root)
    return sorted(root.glob("sub-*/ses-*/eeg/*_eeg.edf"))


def write_trial_cache(
    out_path: Path,
    trials: list[np.ndarray],
    texts: list[str],
    lengths: list[int],
    onsets: list[int],
    meta: dict[str, object],
) -> None:
    """Persist ragged trials as one flat array plus per-trial offsets.

    Trials differ in length, so they cannot be stacked into a dense array; the
    offsets array reconstructs each trial losslessly.
    """
    flat = np.concatenate([t.astype(np.float32) for t in trials], axis=1)
    offsets = np.zeros(len(trials) + 1, dtype=np.int64)
    np.cumsum([t.shape[1] for t in trials], out=offsets[1:])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        flat=flat,
        offsets=offsets,
        n_channels=np.array(trials[0].shape[0]),
        texts=np.array(texts, dtype=object),
        lengths=np.array(lengths, dtype=np.int64),
        onsets=np.array(onsets, dtype=np.int64),
        **meta,
    )
