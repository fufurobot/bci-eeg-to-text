"""Registries for datasets and experiments, so the CLI stays stable.

The project spans several corpora and will span several tasks. Hard-coding a
flag per dataset or a subcommand per experiment means every addition edits the
CLI and risks drift, so both are addressed **by name** through a registry:

    bci dataset list | info NAME | prepare NAME
    bci experiment list | run NAME

Registering a new corpus or a new EEG task is a matter of adding an entry here;
it then appears in ``list`` and becomes runnable with no CLI change.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


class RegistryError(RuntimeError):
    """An unknown or duplicate registry name."""


@dataclass(frozen=True)
class DatasetSpec:
    """What the tooling needs to know about a corpus."""

    name: str
    description: str
    openneuro_id: str
    #: The task this corpus supports, or ``*`` when it is task-agnostic.
    task: str = "eeg2text"
    #: Finest text unit available, matching the leakage modes.
    text_unit: str = "row"
    #: Default subdirectory inside the data root.
    subdir: str = ""
    #: Free-form notes shown by ``bci dataset info``.
    notes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentSpec:
    """What the tooling needs to know about one benchmark row."""

    name: str
    description: str
    task: str = "eeg2text"
    #: Whether the row can be trained here, versus only probed.
    trains_in_process: bool = False
    #: Whether it needs weights downloaded separately.
    needs_external_weights: bool = False
    #: Where to find those weights, for the error message.
    weights_hint: str = ""


_DATASETS: dict[str, Callable[[], DatasetSpec]] = {}
_EXPERIMENTS: dict[str, Callable[[], ExperimentSpec]] = {}


def register_dataset(factory: Callable[[], DatasetSpec] | None = None, *, name: str | None = None):
    """Register a corpus. Usable as a decorator or with an explicit name."""

    def _register(fn: Callable[[], DatasetSpec]):
        spec = fn()
        key = name or spec.name
        if key in _DATASETS:
            raise RegistryError(f"dataset {key!r} is already registered")
        _DATASETS[key] = fn
        return fn

    return _register if factory is None else _register(factory)


def register_experiment(
    factory: Callable[[], ExperimentSpec] | None = None, *, name: str | None = None
):
    """Register an experiment. Usable as a decorator or with an explicit name."""

    def _register(fn: Callable[[], ExperimentSpec]):
        spec = fn()
        key = name or spec.name
        if key in _EXPERIMENTS:
            raise RegistryError(f"experiment {key!r} is already registered")
        _EXPERIMENTS[key] = fn
        return fn

    return _register if factory is None else _register(factory)


def get_dataset(name: str) -> DatasetSpec:
    """Look up a corpus, listing what exists when the name is unknown."""
    try:
        return _DATASETS[name]()
    except KeyError:
        raise RegistryError(
            f"unknown dataset {name!r}; available: {', '.join(sorted(_DATASETS))}"
        ) from None


def get_experiment(name: str) -> ExperimentSpec:
    try:
        return _EXPERIMENTS[name]()
    except KeyError:
        raise RegistryError(
            f"unknown experiment {name!r}; available: {', '.join(sorted(_EXPERIMENTS))}"
        ) from None


def list_datasets(*, task: str | None = None) -> list[str]:
    names = []
    for key, factory in _DATASETS.items():
        spec = factory()
        if task is None or spec.task in (task, "*"):
            names.append(key)
    return sorted(names)


def list_experiments(*, task: str | None = None) -> list[str]:
    names = []
    for key, factory in _EXPERIMENTS.items():
        spec = factory()
        if task is None or spec.task == task:
            names.append(key)
    return sorted(names)


# --------------------------------------------------------------------------
# Built-in corpora. Both are already wired end to end elsewhere in the package;
# this registry is the single place the CLI consults for names and metadata.
# --------------------------------------------------------------------------


@register_dataset
def _chineseeeg() -> DatasetSpec:
    return DatasetSpec(
        name="chineseeeg",
        description="ChineseEEG: silent reading of Chinese novels, 128 channels",
        openneuro_id="ds004952",
        task="eeg2text",
        # Rows are verbatim substrings of the source novels, so paragraph and
        # article leakage modes are enforceable.
        text_unit="article",
        subdir="ds004952",
        notes={"subjects": "10 (5F/5M)", "rate": "256 Hz (decimated on read)"},
    )


@register_dataset
def _chisco() -> DatasetSpec:
    return DatasetSpec(
        name="chisco",
        description="Chisco: imagined speech of Chinese sentences, reward-pulse aligned",
        openneuro_id="ds005170",
        task="eeg2text",
        # Stimuli are standalone sentences with no surrounding document, so
        # text-grouping beyond the row is not meaningful.
        text_unit="row",
        subdir="ds005170",
        notes={"subjects": "4 usable (2F/2M)", "evoked": "output-only reward pulses"},
    )


# --------------------------------------------------------------------------
# Built-in experiments.
# --------------------------------------------------------------------------


@register_experiment
def _echo() -> ExperimentSpec:
    return ExperimentSpec(
        name="echo",
        description="ECHO: FAST encoder, projection, Whisper decoder (seq2seq)",
        trains_in_process=False,
        needs_external_weights=True,
        weights_hint="~/ECHO/checkpoints/ECHO.ckpt",
    )


@register_experiment
def _thought2text() -> ExperimentSpec:
    return ExperimentSpec(
        name="thought2text",
        description="Thought2Text: ChannelNet EEG encoder plus an LLM (Mistral-7B)",
        trains_in_process=False,
        needs_external_weights=True,
        weights_hint="~/hf-models/Mistral-7B-v0.3 plus LoRA adapters",
    )


@register_experiment
def _conv_transformer() -> ExperimentSpec:
    return ExperimentSpec(
        name="conv-transformer",
        description="ours: conv front-end + transformer encoder, CTC, one pass",
        trains_in_process=True,
    )


@register_experiment
def _conv_bert() -> ExperimentSpec:
    return ExperimentSpec(
        name="conv-bert",
        description="ours: conv front-end + BERT encoder + CTC (fine-tuned here)",
        trains_in_process=True,
        needs_external_weights=False,
    )


@register_experiment
def _neurolm() -> ExperimentSpec:
    return ExperimentSpec(
        name="neurolm",
        description="NeuroLM: NeuralTransformerVQ plus adapter head, CTC",
        trains_in_process=False,
        needs_external_weights=True,
        weights_hint="~/hf-models/NeuroLM/checkpoints/NeuroLM-B.pt",
    )
