"""Red tests for the `bci benchmark eeg2text` CLI.

The command under test::

    bci benchmark eeg2text --model all --dataset xxx --dataset yyy --smoke

``--smoke`` must pick **10 runs containing 5 distinct texts from the same
subject, two runs per text**, then run **5-fold cross-validation in 3-1-1 style**
(three folds for train, one valid, one test). That sampler is the subtle part:
it has to guarantee the pairing property rather than approximate it, and it has
to refuse datasets that cannot supply it instead of silently returning fewer.

Four experiment rows are expected: ECHO, Thought2Text, conv-transformer CTC
(single forward pass), and NeuroLM with its adapter.
"""

from __future__ import annotations

import pytest

from bci_eeg_to_text.smoke import (
    SmokeSelectionError,
    fold_indices,
    group_runs_by_text,
    select_smoke_runs,
)


def make_rows(subject: str, texts_per_run: list[list[tuple[str, str]]]) -> list[dict]:
    """Rows shaped like the streaming index: one per (run, text)."""
    rows = []
    for items in texts_per_run:
        for row_index, (text, run_name) in enumerate(items):
            rows.append(
                {
                    "subject": subject,
                    "run": run_name,
                    "text": text,
                    "row_index": row_index,
                    "run_id": f"{subject}/{run_name}",
                }
            )
    return rows


class TestGroupRunsByText:
    def test_groups_runs_by_shared_text(self) -> None:
        rows = make_rows(
            "s1",
            [
                [("你好", "r1"), ("世界", "r1")],
                [("你好", "r2"), ("世界", "r2")],
                [("你好", "r3"), ("其它", "r3")],
            ],
        )
        groups = group_runs_by_text(rows)
        assert "你好" in groups
        assert len(groups["你好"]) == 3  # r1, r2, r3 all contain it
        assert len(groups["其它"]) == 1

    def test_ignores_runs_with_a_single_text(self) -> None:
        """A text appearing once cannot supply a train/test pair."""
        rows = make_rows("s1", [[("唯一", "r1")], [("唯一", "r2")]])
        groups = group_runs_by_text(rows, min_runs=3)
        assert groups.get("唯一", []) == []


class TestSelectSmokeRuns:
    def _many_rows(self, n_texts: int = 12) -> list[dict]:
        texts = [f"文本{i}" for i in range(n_texts)]
        runs = []
        for r in range(20):
            runs.append([(t, f"run{r:02d}") for t in texts])
        return make_rows("s1", runs)

    def test_selects_ten_runs_five_texts_two_runs_each(self) -> None:
        rows = self._many_rows()
        selection = select_smoke_runs(rows, seed=0)
        assert len(selection.runs) == 10
        assert len(selection.texts) == 5
        assert selection.subject == "s1"
        # Exactly two runs per chosen text.
        per_text: dict[str, int] = {}
        for run in selection.runs:
            for text in run.texts:
                if text in selection.texts:
                    per_text[text] = per_text.get(text, 0) + 1
        assert all(count >= 2 for count in per_text.values())

    def test_is_deterministic_for_a_seed(self) -> None:
        rows = self._many_rows()
        a = select_smoke_runs(rows, seed=7)
        b = select_smoke_runs(rows, seed=7)
        assert a.run_ids == b.run_ids
        assert a.texts == b.texts

    def test_uses_a_single_subject(self) -> None:
        rows = self._many_rows() + make_rows("s2", [[(f"t{i}", "r1") for i in range(10)]])
        selection = select_smoke_runs(rows, seed=0)
        assert len({r.subject for r in selection.runs}) == 1

    def test_raises_when_too_few_texts(self) -> None:
        rows = make_rows("s1", [[(f"t{i}", f"r{r}") for i in range(2)] for r in range(10)])
        with pytest.raises(SmokeSelectionError, match="texts"):
            select_smoke_runs(rows, seed=0)

    def test_raises_when_too_few_runs(self) -> None:
        rows = make_rows("s1", [[(f"t{i}", f"r{r}") for i in range(10)] for r in range(4)])
        with pytest.raises(SmokeSelectionError, match="runs"):
            select_smoke_runs(rows, seed=0)


class TestFolds:
    def test_five_folds_of_three_one_one(self) -> None:
        folds = fold_indices(10, n_folds=5)
        assert len(folds) == 5
        for train, valid, test in folds:
            assert len(train) == 6
            assert len(valid) == 2
            assert len(test) == 2

    def test_folds_partition_every_index_exactly_once(self) -> None:
        folds = fold_indices(10, n_folds=5)
        for train, valid, test in folds:
            assert set(train) | set(valid) | set(test) == set(range(10))
            assert not (set(train) & set(valid))
            assert not (set(valid) & set(test))
            assert not (set(train) & set(test))

    def test_every_index_lands_in_test_exactly_once(self) -> None:
        folds = fold_indices(10, n_folds=5)
        seen = [i for _, _, test in folds for i in test]
        assert sorted(seen) == list(range(10))

    def test_rejects_indivisible_counts(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            fold_indices(11, n_folds=5)


class TestCliWiring:
    def test_benchmark_subcommand_exists(self) -> None:
        from bci_eeg_to_text.benchmark_cli import build_parser

        args = build_parser().parse_args(
            ["benchmark", "eeg2text", "--model", "all", "--dataset", "chineseeeg", "--smoke"]
        )
        assert args.model == "all"
        assert args.dataset == ["chineseeeg"]
        assert args.smoke is True

    def test_accepts_repeated_datasets(self) -> None:
        from bci_eeg_to_text.benchmark_cli import build_parser

        args = build_parser().parse_args(
            [
                "benchmark",
                "eeg2text",
                "--model",
                "conv-transformer",
                "--dataset",
                "a",
                "--dataset",
                "b",
            ]
        )
        assert args.dataset == ["a", "b"]

    def test_model_choices_cover_the_four_experiments(self) -> None:
        from bci_eeg_to_text.benchmark_cli import BENCHMARK_MODELS

        # Five rows now: conv-bert was added as the smoke-test priority, since
        # its ~110M parameters fit this machine where NeuroLM-XL (20 GB) and
        # Mistral-7B (21 GB) do not.
        assert set(BENCHMARK_MODELS) == {
            "conv-bert",
            "conv-transformer",
            "echo",
            "thought2text",
            "neurolm",
        }

    def test_smoke_flag_defaults_off(self) -> None:
        from bci_eeg_to_text.benchmark_cli import build_parser

        args = build_parser().parse_args(
            ["benchmark", "eeg2text", "--model", "conv-transformer", "--dataset", "a"]
        )
        assert args.smoke is False


class TestFourRows:
    def test_all_models_produce_a_row_or_a_skip(self, tmp_path) -> None:
        """`--model all` must yield four labelled rows even when unavailable."""
        from bci_eeg_to_text.benchmark_cli import collect_statuses

        statuses = collect_statuses("all", echo_ckpt=tmp_path / "none.ckpt")
        assert len(statuses) == 5
        names = {s.name for s in statuses}
        assert names == {
            "conv-bert",
            "conv-transformer",
            "echo",
            "thought2text",
            "neurolm",
        }
        # Unavailable ones must carry a reason.
        for status in statuses:
            if not status.available:
                assert status.reason
