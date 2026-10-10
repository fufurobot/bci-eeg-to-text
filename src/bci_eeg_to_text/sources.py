"""Dataset sources: one interface, several EEG-text corpora.

The point of this module is that **the model does not care which dataset the
data came from**. Both supported corpora are reduced to the same thing: a list
of :class:`~bci_eeg_to_text.eeg.Segment` records pairing an EEG sample range
with the text that was being read or imagined.

They differ only in how that pairing is obtained:

* ``ds004952`` (ChineseEEG) ships stimulus markers (``ROWS``/``ROWE``), so the
  pairing is *read* from the events.
* ``ds005170`` (Chisco) is output-only and has no stimulus triggers, so the
  pairing is *recovered* from the uniform reward pulses (see issue #1).

Registration happens through :func:`register_source` so the CLI can enumerate
datasets by name without importing them eagerly.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from bci_eeg_to_text.splits import Participant


@dataclass(frozen=True)
class RunInfo:
    """One recording that can yield segments."""

    subject_id: str
    dataset: str
    run: str
    path: Path
    #: Session label when the dataset uses sessions (e.g. "ses-LittlePrince").
    session: str | None = None

    @property
    def namespaced_subject(self) -> str:
        return f"{self.dataset}:{self.subject_id}"


@dataclass(frozen=True)
class ExtractionReport:
    """Outcome of extracting one run."""

    run: RunInfo
    n_segments: int
    skipped_reason: str | None = None
    details: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.skipped_reason is None


@runtime_checkable
class DatasetSource(Protocol):
    """A corpus of ``(EEG segment, text)`` pairs.

    Implementations must be read-only with respect to the data: nothing under
    the dataset root is ever written.
    """

    #: Short stable name used on the command line.
    name: str

    def root(self) -> Path:
        """Filesystem root of the dataset."""

    def participants(self) -> list[Participant]:
        """Subjects with usable data, including sex for balanced splitting."""

    def runs(self) -> list[RunInfo]:
        """Every usable recording, sorted deterministically."""

    def extract(self, run: RunInfo, out_path: Path, **options: object) -> ExtractionReport:
        """Read one run and write a cached trial file."""

    def describe(self) -> str:
        """One-line human description for CLI output."""


_REGISTRY: dict[str, type] = {}


def register_source(cls: type) -> type:
    """Register a :class:`DatasetSource` implementation by its ``name``."""
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name:
        raise ValueError(f"{cls.__name__} must define a non-empty 'name'")
    _REGISTRY[name] = cls
    return cls


def available_sources() -> list[str]:
    """Names of all registered sources, sorted."""
    _ensure_loaded()
    return sorted(_REGISTRY)


def get_source(name: str, *args: object, **kwargs: object) -> DatasetSource:
    """Instantiate a registered source by name."""
    _ensure_loaded()
    if name not in _REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](*args, **kwargs)


def _ensure_loaded() -> None:
    """Import the built-in sources so they register themselves."""
    if _REGISTRY:
        return
    from bci_eeg_to_text import chineseeeg, chisco  # noqa: F401


def subjects_for_split(sources: Iterable[DatasetSource]) -> list[Participant]:
    """Collect participants from several sources, namespacing their ids.

    Both datasets use bare ``sub-NN`` identifiers, so ids must be namespaced or
    ``sub-04`` from each would collide in a merged split.
    """
    out: list[Participant] = []
    for source in sources:
        for participant in source.participants():
            out.append(
                Participant(
                    subject_id=f"{source.name}:{participant.short_id}",
                    dataset=source.name,
                    sex=participant.sex,
                )
            )
    return out


def load_participants_tsv(path: str | Path) -> list[Participant]:
    """Read a BIDS ``participants.tsv`` into :class:`Participant` records.

    Missing or ``n/a`` sex values become ``None``, which disables balancing for
    those subjects rather than inventing a value.
    """
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8-sig").strip().splitlines()
    if not lines:
        return []
    header = lines[0].split("\t")
    try:
        id_col = header.index("participant_id")
    except ValueError as exc:
        raise ValueError(f"{path.name}: missing participant_id column") from exc
    sex_col = header.index("sex") if "sex" in header else None

    out: list[Participant] = []
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) <= id_col or not fields[id_col].strip():
            continue
        sex: str | None = None
        if sex_col is not None and sex_col < len(fields):
            raw = fields[sex_col].strip().upper()
            if raw and raw not in {"N/A", "NA", "UNKNOWN", "-"}:
                sex = raw[0]
        out.append(Participant(subject_id=fields[id_col].strip(), sex=sex))
    return out


def usable_subjects(
    participants: Sequence[Participant],
    runs: Sequence[RunInfo],
) -> list[Participant]:
    """Keep only participants that actually have usable runs.

    A participant listed in ``participants.tsv`` may have no data locally (for
    example when only some subjects were downloaded), and including them would
    put an empty subject in a split.
    """
    have = {r.subject_id for r in runs}
    return [p for p in participants if p.short_id in have]
