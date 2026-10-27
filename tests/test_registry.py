"""Red tests for the unified `bci dataset` / `bci experiment` entrypoints.

The project now spans several corpora (ChineseEEG, Chisco) and several tasks
(EEG-to-text today, more later). Commands growing per dataset and per task would
duplicate flags and drift apart, so both are addressed by **name** through one
registry each:

* ``bci dataset list|info|prepare`` -- what corpora exist and how to stage them.
* ``bci experiment run|list`` -- what experiments exist and how to launch one.

Registries must be extensible without touching the CLI: registering a new
dataset or task has to be enough for it to appear in ``list`` and be runnable.
"""

from __future__ import annotations

import pytest

from bci_eeg_to_text.registry import (
    DatasetSpec,
    ExperimentSpec,
    RegistryError,
    get_dataset,
    get_experiment,
    list_datasets,
    list_experiments,
    register_dataset,
    register_experiment,
)


class TestDatasetRegistry:
    def test_known_datasets_are_listed(self) -> None:
        names = set(list_datasets())
        assert {"chineseeeg", "chisco"} <= names

    def test_get_dataset_returns_a_spec(self) -> None:
        spec = get_dataset("chineseeeg")
        assert spec.name == "chineseeeg"
        assert spec.openneuro_id
        assert spec.task in {"eeg2text", "*"}

    def test_unknown_dataset_raises_with_suggestions(self) -> None:
        with pytest.raises(RegistryError, match="unknown dataset"):
            get_dataset("no-such-dataset")

    def test_get_dataset_error_lists_available(self) -> None:
        with pytest.raises(RegistryError) as exc:
            get_dataset("nope")
        assert "chineseeeg" in str(exc.value)

    def test_registering_a_dataset_makes_it_visible(self) -> None:
        """Adding a corpus must not require editing the CLI."""

        @register_dataset
        def _new_corpus() -> DatasetSpec:
            return DatasetSpec(
                name="toy-corpus",
                description="a corpus added at runtime for this test",
                openneuro_id="ds000000",
                task="eeg2text",
                text_unit="row",
            )

        assert "toy-corpus" in list_datasets()
        assert get_dataset("toy-corpus").openneuro_id == "ds000000"

    def test_duplicate_registration_is_rejected(self) -> None:
        with pytest.raises(RegistryError, match="already registered"):
            register_dataset(
                lambda: DatasetSpec(
                    name="chineseeeg",
                    description="dup",
                    openneuro_id="x",
                    task="eeg2text",
                    text_unit="row",
                ),
                name="chineseeeg",
            )


class TestExperimentRegistry:
    def test_known_experiments_are_listed(self) -> None:
        names = set(list_experiments())
        assert {"echo", "thought2text", "conv-transformer", "neurolm"} <= names

    def test_get_experiment_returns_a_spec(self) -> None:
        spec = get_experiment("conv-transformer")
        assert spec.name == "conv-transformer"
        assert spec.trains_in_process is True

    def test_readme_says_unavailable_for_weights_dependent_rows(self) -> None:
        """Rows needing external weights declare it rather than pretending."""
        spec = get_experiment("neurolm")
        assert spec.needs_external_weights is True

    def test_unknown_experiment_raises(self) -> None:
        with pytest.raises(RegistryError, match="unknown experiment"):
            get_experiment("nope")

    def test_registering_an_experiment_makes_it_visible(self) -> None:
        @register_experiment
        def _toy_task() -> ExperimentSpec:
            return ExperimentSpec(
                name="toy-task",
                description="another EEG task added later",
                task="eeg2text",
                trains_in_process=False,
            )

        assert "toy-task" in list_experiments()

    def test_experiments_can_be_filtered_by_task(self) -> None:
        """Future EEG tasks must not pollute the eeg2text list."""
        register_experiment(
            lambda: ExperimentSpec(
                name="toy-eeg2image",
                description="different task",
                task="eeg2image",
                trains_in_process=True,
            ),
            name="toy-eeg2image",
        )
        assert "toy-eeg2image" not in list_experiments(task="eeg2text")
        assert "toy-eeg2image" in list_experiments(task="eeg2image")


class TestCliSurface:
    def test_dataset_subcommands_parse(self) -> None:
        from bci_eeg_to_text.cli import build_parser

        args = build_parser().parse_args(["dataset", "list"])
        assert args.command == "dataset"

    def test_experiment_subcommands_parse(self) -> None:
        from bci_eeg_to_text.cli import build_parser

        args = build_parser().parse_args(["experiment", "list"])
        assert args.command == "experiment"

    def test_experiment_run_accepts_dataset_and_mode(self) -> None:
        from bci_eeg_to_text.cli import build_parser

        args = build_parser().parse_args(
            [
                "experiment",
                "run",
                "conv-transformer",
                "--dataset",
                "chineseeeg",
                "--training-mode",
                "cpu",
            ]
        )
        assert args.name == "conv-transformer"
        assert args.training_mode == "cpu"

    def test_training_mode_flag_covers_three_backends(self) -> None:
        from bci_eeg_to_text.cli import build_parser

        for mode in ("cpu", "gpu", "ssh"):
            args = build_parser().parse_args(
                ["experiment", "run", "echo", "--dataset", "chineseeeg", "--training-mode", mode]
            )
            assert args.training_mode == mode

    def test_strategy_flag_is_exposed(self) -> None:
        from bci_eeg_to_text.cli import build_parser

        args = build_parser().parse_args(
            ["experiment", "run", "echo", "--dataset", "chineseeeg", "--strategy", "fsdp"]
        )
        assert args.strategy == "fsdp"

    def test_topology_flag_is_exposed(self) -> None:
        from bci_eeg_to_text.cli import build_parser

        args = build_parser().parse_args(
            [
                "experiment",
                "run",
                "echo",
                "--dataset",
                "chineseeeg",
                "--training-mode",
                "ssh",
                "--hosts",
                "j:2|w:4",
            ]
        )
        assert args.hosts == "j:2|w:4"


class TestUnifiedEntrypointDispatch:
    def test_dataset_list_runs_and_prints(self, capsys) -> None:
        from bci_eeg_to_text.cli import main

        code = main(["dataset", "list"])
        out = capsys.readouterr().out
        assert code == 0
        assert "chineseeeg" in out

    def test_experiment_list_runs_and_prints(self, capsys) -> None:
        from bci_eeg_to_text.cli import main

        code = main(["experiment", "list"])
        out = capsys.readouterr().out
        assert code == 0
        assert "conv-transformer" in out

    def test_unknown_dataset_prints_a_clean_error(self, capsys) -> None:
        from bci_eeg_to_text.cli import main

        code = main(["dataset", "info", "nope"])
        err = capsys.readouterr().err
        assert code == 1
        assert "unknown dataset" in err
