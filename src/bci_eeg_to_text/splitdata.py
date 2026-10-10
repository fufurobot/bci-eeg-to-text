"""Turning a materialised split tree into training-ready datasets.

This is the bridge that makes `--split` meaningful: training reads the *split
tree*, not an ad-hoc directory of cached arrays. A part's recordings are
extracted on demand and cached keyed by the resolved source path, so the cache
is an implementation detail rather than an input the user must prepare.

The distinction matters because the previous design globbed every ``.npz`` in a
cache directory. That silently mixed parts: with a merged corpus a test subject
could be trained on, and the tokenizer was fitted on evaluation text. Both are
leakage, and both looked like good results.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from bci_eeg_to_text.splittree import SplitTree


class BuildError(RuntimeError):
    """Raised when training data cannot be built from a split."""


@dataclass
class TrainingData:
    """Datasets and vocabulary derived from a split tree."""

    train: object
    valid: object | None
    tokenizer: object
    n_channels: int
    valid_part: str | None
    n_train_files: int
    n_valid_files: int
    per_part: dict[str, int]


def _source_for_recording(recording: Path):
    """Pick the dataset source able to read ``recording``."""
    from bci_eeg_to_text.sources import get_source

    text = str(recording)
    if "ds005170" in text:
        return get_source("chisco"), "chisco"
    if "ds004952" in text:
        return get_source("chineseeeg"), "chineseeeg"
    return None, None


def _find_run_info(source, recording: Path):
    """Locate the :class:`RunInfo` whose file matches ``recording``."""
    from bci_eeg_to_text.eeg import RunId

    resolved = recording.resolve()
    # Comparing resolved paths is the only reliable match: os.path.samefile
    # fails across some mounts, and names are ambiguous (run numbers repeat
    # across sessions).
    for run in source.runs():
        try:
            if run.path.resolve() == resolved:
                return run
        except OSError:
            continue
    if "ds005170" in str(resolved) and resolved.suffix == ".edf":
        run_id = RunId.parse(resolved.name)
        padded = f"run-{run_id.run:03d}"
        for run in source.runs():
            if run.subject_id == run_id.subject and run.run == padded:
                return run
    return None


def extract_part(
    tree: SplitTree,
    part: str,
    cache_dir: Path,
    *,
    decimate: int = 8,
    verbose: bool = True,
) -> tuple[list[Path], list[str]]:
    """Extract every recording of ``part``; returns ``(cache paths, warnings)``.

    Cache filenames embed a hash of the resolved source path, so re-running is
    cheap and two different recordings can never share a cache file.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    warnings: list[str] = []

    for entry in tree.parts.get(part, []):
        for recording in entry.run_files:
            source, name = _source_for_recording(recording)
            if source is None:
                warnings.append(f"{recording.name}: unrecognised dataset")
                continue
            run_info = _find_run_info(source, recording)
            if run_info is None:
                warnings.append(f"{recording.name}: not a run known to {name}")
                continue

            digest = hashlib.sha256(str(recording.resolve()).encode()).hexdigest()[:12]
            out = cache_dir / f"{part}_{recording.stem[:60]}_{digest}.npz"
            if out.exists():
                written.append(out)
                continue
            try:
                source.extract(run_info, out, decimate=decimate)
            except Exception as exc:  # noqa: BLE001 - report and continue
                warnings.append(f"{recording.name}: {exc}")
                continue
            written.append(out)
            if verbose:
                print(f"  extracted {part}/{recording.name}", flush=True)

    return written, warnings


def build_training_data(
    tree: SplitTree,
    *,
    n_samples: int,
    cache_dir: Path,
    decimate: int = 8,
    verbose: bool = True,
) -> TrainingData:
    """Extract the split tree and assemble datasets plus a tokenizer.

    The tokenizer is fitted on **training text only**. Fitting it on the
    evaluation split would put its characters in the label space, which is a
    subtle form of leakage.
    """
    import numpy as np

    from bci_eeg_to_text.data import TrialDataset
    from bci_eeg_to_text.tokenizer import CharTokenizer

    extracted: dict[str, list[Path]] = {}
    warnings: list[str] = []
    for part in tree.parts:
        paths, part_warnings = extract_part(
            tree, part, cache_dir / part, decimate=decimate, verbose=verbose
        )
        extracted[part] = paths
        warnings.extend(part_warnings)

    if not extracted.get("train"):
        raise BuildError(
            "no training recordings could be extracted from the split. "
            "Check that the dataset is mounted and the links resolve "
            "(the warnings above say which recordings failed)."
        )

    texts: list[str] = []
    n_channels: int | None = None
    for path in extracted["train"]:
        with np.load(path, allow_pickle=True) as data:
            texts.extend(str(t) for t in data["texts"])
            if n_channels is None:
                n_channels = int(data["n_channels"])
    if n_channels is None:
        raise BuildError("training caches contain no trials")

    tokenizer = CharTokenizer.build(texts)
    train = TrialDataset(extracted["train"], tokenizer, n_samples=n_samples)

    # Prefer a real validation split; fall back to test only if there is none.
    valid_part = "valid" if extracted.get("valid") else ("test" if extracted.get("test") else None)
    valid = (
        TrialDataset(extracted[valid_part], tokenizer, n_samples=n_samples)
        if valid_part and extracted[valid_part]
        else None
    )

    return TrainingData(
        train=train,
        valid=valid,
        tokenizer=tokenizer,
        n_channels=n_channels,
        valid_part=valid_part,
        n_train_files=len(extracted["train"]),
        n_valid_files=len(extracted.get(valid_part, [])) if valid_part else 0,
        per_part={part: len(paths) for part, paths in extracted.items()},
    )
