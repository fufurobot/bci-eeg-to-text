"""Reading Chisco (ds005170) EDF runs and recovering trial boundaries.

The dataset stores **output-only** EEG: the marker channel carries a single
uniform reward pulse per imagined sentence and no stimulus-onset triggers.
Alignment therefore depends on recovering those pulses, which is what this
module does.

See issue #1 for the evidence behind this design.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Value of the uniform reward pulse in the Trigger channel.
REWARD_PULSE = 65380

#: Channels that are recorded alongside the EEG but are not EEG.
NON_EEG_CHANNELS = frozenset({"VEO", "HEO", "EKG", "EMG", "Trigger", "EDF Annotations"})

TRIGGER_CHANNEL = "Trigger"

#: Expected EEG channel count for ds005170 (EGI 128 layout).
EXPECTED_EEG_CHANNELS = 128

_RUN_RE = re.compile(
    r"^(?P<subject>sub-\d+)_ses-(?P<session>\d+)_task-(?P<task>[\w]+)_run-(?P<run>\d+)_eeg\.edf$"
)


class AlignmentError(RuntimeError):
    """Raised when EEG segments cannot be aligned to target sentences."""


class MissingDataError(RuntimeError):
    """Raised when an EDF file is a git-annex pointer rather than real data.

    This dataset is distributed via DataLad/git-annex. Files whose content was
    never fetched are small text pointers, not EDF. Reading one produces a
    confusing "not EDF compliant" error, so it is detected explicitly.
    """


#: Anything at most this large could be an annex pointer rather than EDF.
ANNEX_POINTER_MAX_BYTES = 4096
_ANNEX_POINTER_RE = re.compile(rb"^/annex/objects/\S+")


def is_annex_pointer(path: str | Path) -> bool:
    """Return True if ``path`` is an unfetched git-annex pointer stub."""
    path = Path(path)
    try:
        if path.stat().st_size > ANNEX_POINTER_MAX_BYTES:
            return False
        with path.open("rb") as handle:
            return bool(_ANNEX_POINTER_RE.match(handle.read(64)))
    except OSError:
        return False


def require_real_data(path: str | Path) -> None:
    """Raise :class:`MissingDataError` if ``path`` is not usable EDF data.

    Catches both unfetched git-annex pointers and truncated partial downloads.
    """
    path = Path(path)
    if is_annex_pointer(path):
        raise MissingDataError(
            f"{path.name} is an unfetched git-annex pointer, not EDF data. "
            "Fetch it with `datalad get` / `git annex get` from the dataset remote."
        )
    if not is_readable_edf(path):
        raise MissingDataError(
            f"{path.name} cannot be read as EDF. It is most likely an incomplete "
            f"download ({path.stat().st_size} bytes on disk). Fetch it again with "
            "`datalad get` / `git annex get`."
        )


def is_readable_edf(path: str | Path) -> bool:
    """Return True if pyedflib can open the file and read its header.

    This is the authoritative check for a usable recording: it is the same
    library used for reading, so it agrees exactly with what training will
    experience. Truncated downloads fail here with "not EDF compliant
    (Filesize)", and annex pointers fail because they contain no EDF header.
    """
    import pyedflib

    try:
        with pyedflib.EdfReader(str(path)) as reader:
            labels = reader.getSignalLabels()
            return bool(labels) and TRIGGER_CHANNEL in labels
    except Exception:  # noqa: BLE001 - any failure means "not usable"
        return False


def usable_runs(root: str | Path) -> list[Path]:
    """List runs that hold real, complete EDF data.

    Skips unfetched annex pointers and truncated partial downloads. The Chisco
    release ships 46 pointers and several truncated files, so training must
    plan around the runs that actually exist locally.
    """
    return [p for p in iter_runs(root) if not is_incomplete(p)]


def is_incomplete(path: str | Path) -> bool:
    """Return True if ``path`` is an annex pointer or an unreadable download."""
    path = Path(path)
    if is_annex_pointer(path):
        return True
    return not is_readable_edf(path)


@dataclass(frozen=True)
class RunId:
    """Identity of a single BIDS EEG run."""

    subject: str
    session: int
    task: str
    run: int

    @property
    def key(self) -> str:
        return f"{self.subject}_ses-{self.session:02d}_task-{self.task}_run-{self.run:03d}"

    @classmethod
    def parse(cls, filename: str) -> RunId:
        """Parse a BIDS EEG filename into a :class:`RunId`."""
        m = _RUN_RE.match(Path(filename).name)
        if not m:
            raise ValueError(f"not a BIDS EEG filename: {filename!r}")
        return cls(
            subject=m.group("subject"),
            session=int(m.group("session")),
            task=m.group("task"),
            run=int(m.group("run")),
        )


@dataclass(frozen=True)
class RunMeta:
    """Header-level metadata for one EDF run, without loading the signal."""

    run_id: RunId
    path: Path
    n_eeg_channels: int
    sample_rate_hz: float
    n_samples: int
    channel_names: tuple[str, ...]

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sample_rate_hz


@dataclass(frozen=True)
class Segment:
    """A single imagined-sentence trial: a sample range plus its target text."""

    index: int
    start_sample: int
    stop_sample: int
    text: str

    @property
    def n_samples(self) -> int:
        return self.stop_sample - self.start_sample


def read_signals(
    path: str | Path, channels: Sequence[str] | None = None
) -> tuple[np.ndarray, float]:
    """Read selected channels from an EDF file.

    Returns ``(data, sample_rate_hz)`` where ``data`` has shape
    ``(n_channels, n_samples)``. Only the requested channels are loaded, which
    matters because the raw files are hundreds of megabytes each.
    """
    import pyedflib

    require_real_data(path)
    with pyedflib.EdfReader(str(path)) as reader:
        available = list(reader.getSignalLabels())
        names = available if channels is None else list(channels)
        missing = [c for c in names if c not in available]
        if missing:
            raise KeyError(f"channels not present in {Path(path).name}: {missing}")
        sample_rate = float(reader.getSampleFrequency(available.index(names[0])))
        data = np.stack([reader.readSignal(available.index(c)) for c in names])
    return data, sample_rate


def read_metadata(path: str | Path) -> RunMeta:
    """Read header metadata for an EDF run without loading its signal."""
    import pyedflib

    path = Path(path)
    require_real_data(path)
    with pyedflib.EdfReader(str(path)) as reader:
        labels = list(reader.getSignalLabels())
        eeg = [c for c in labels if c not in NON_EEG_CHANNELS]
        return RunMeta(
            run_id=RunId.parse(path.name),
            path=path,
            n_eeg_channels=len(eeg),
            sample_rate_hz=float(reader.getSampleFrequency(0)),
            n_samples=int(reader.getNSamples()[0]),
            channel_names=tuple(labels),
        )


def find_pulses(
    trigger: np.ndarray,
    *,
    pulse_value: int = REWARD_PULSE,
) -> np.ndarray:
    """Return the onset sample index of every reward pulse in ``trigger``.

    A pulse is a run of consecutive samples equal to ``pulse_value``; the
    returned index is the first sample of that run, i.e. the pulse onset.
    """
    trigger = np.asarray(trigger)
    if trigger.ndim != 1:
        raise ValueError(f"trigger must be 1-D, got shape {trigger.shape}")
    is_pulse = trigger == pulse_value
    if not is_pulse.any():
        return np.empty(0, dtype=np.int64)
    # A rising edge is a pulse sample whose predecessor is not a pulse.
    padded = np.concatenate(([False], is_pulse))
    rising = is_pulse & ~padded[:-1]
    return np.flatnonzero(rising).astype(np.int64)


def segment_bounds(
    onsets: Sequence[int],
    *,
    n_samples: int,
    pre_samples: int = 0,
    post_samples: int | None = None,
) -> list[tuple[int, int]]:
    """Turn pulse onsets into ``(start, stop)`` sample ranges.

    Each segment starts ``pre_samples`` before its pulse onset and ends either
    ``post_samples`` after the onset, or at the next pulse onset when
    ``post_samples`` is None. The final segment is clipped to ``n_samples``.
    """
    onsets = [int(o) for o in onsets]
    if any(b <= a for a, b in zip(onsets, onsets[1:], strict=False)):
        raise AlignmentError("pulse onsets must be strictly increasing")
    bounds: list[tuple[int, int]] = []
    for i, onset in enumerate(onsets):
        start = onset - pre_samples
        if post_samples is None:
            stop = onsets[i + 1] if i + 1 < len(onsets) else n_samples
        else:
            stop = onset + post_samples
        start = max(0, start)
        stop = min(n_samples, stop)
        bounds.append((start, stop))
    return bounds


def align_segments(
    onsets: Sequence[int],
    texts: Sequence[str],
    *,
    n_samples: int,
    pre_samples: int = 0,
    post_samples: int | None = None,
) -> list[Segment]:
    """Pair pulse onsets with target sentences, failing loudly on mismatch.

    The number of reward pulses must equal the number of target sentences.
    Truncating or padding here would silently train the model on misaligned
    pairs, so a mismatch raises :class:`AlignmentError` instead.
    """
    if len(onsets) != len(texts):
        raise AlignmentError(
            f"reward pulses ({len(onsets)}) != target sentences ({len(texts)}); "
            "refusing to train on misaligned pairs - see issue #1"
        )
    bounds = segment_bounds(
        onsets, n_samples=n_samples, pre_samples=pre_samples, post_samples=post_samples
    )
    segments: list[Segment] = []
    for i, ((start, stop), text) in enumerate(zip(bounds, texts, strict=True)):
        if stop <= start:
            raise AlignmentError(f"segment {i} is empty: [{start}, {stop})")
        segments.append(Segment(index=i, start_sample=start, stop_sample=stop, text=text))
    return segments


def iter_runs(root: str | Path) -> Iterator[Path]:
    """Yield every EDF run under a dataset root, sorted for determinism.

    Includes runs whose files are still annex pointers; use
    :func:`usable_runs` to skip those.
    """
    yield from sorted(Path(root).glob("sub-*/ses-*/eeg/*_eeg.edf"))
