"""Tests for participant-level, sex-balanced splitting."""

from __future__ import annotations

from collections import Counter

import pytest

from bci_eeg_to_text.splits import (
    Participant,
    SplitError,
    _target_counts,
    make_splits,
    validate_subject_disjoint,
)

#: Real demographics of the two datasets.
CHISCO = [  # ds005170, 4 usable subjects (sub-03 is placeholder-only)
    Participant("chisco:sub-01", "chisco", "F"),
    Participant("chisco:sub-02", "chisco", "M"),
    Participant("chisco:sub-04", "chisco", "M"),
    Participant("chisco:sub-05", "chisco", "F"),
]
CHINESEEG = [  # ds004952, 10 subjects
    Participant("chineseeeg:sub-04", "chineseeeg", "F"),
    Participant("chineseeeg:sub-05", "chineseeeg", "F"),
    Participant("chineseeeg:sub-06", "chineseeeg", "F"),
    Participant("chineseeeg:sub-07", "chineseeeg", "F"),
    Participant("chineseeeg:sub-08", "chineseeeg", "F"),
    Participant("chineseeeg:sub-09", "chineseeeg", "M"),
    Participant("chineseeeg:sub-10", "chineseeeg", "M"),
    Participant("chineseeeg:sub-13", "chineseeeg", "M"),
    Participant("chineseeeg:sub-14", "chineseeeg", "M"),
    Participant("chineseeeg:sub-15", "chineseeeg", "M"),
]


class TestTargetCounts:
    def test_three_to_one_to_one_of_ten(self) -> None:
        assert _target_counts(10, {"train": 3, "valid": 1, "test": 1}) == {
            "train": 6,
            "valid": 2,
            "test": 2,
        }

    def test_one_to_one_of_four(self) -> None:
        assert _target_counts(4, {"train": 1, "test": 1}) == {"train": 2, "test": 2}

    def test_counts_always_sum_to_total(self) -> None:
        for n in range(1, 40):
            counts = _target_counts(n, {"train": 3, "valid": 1, "test": 1})
            assert sum(counts.values()) == n

    def test_rejects_zero_ratio_sum(self) -> None:
        with pytest.raises(SplitError, match="sum to a positive"):
            _target_counts(5, {"train": 0})


class TestSubjectLevelSplits:
    def test_chineseeeg_three_one_one(self) -> None:
        split = make_splits(CHINESEEG, ratios={"train": 3, "valid": 1, "test": 1}, seed=0)
        assert {p: len(s) for p, s in split.parts.items()} == {
            "train": 6,
            "valid": 2,
            "test": 2,
        }

    def test_chisco_one_to_one(self) -> None:
        split = make_splits(CHISCO, ratios={"train": 1, "test": 1}, seed=0)
        assert {p: len(s) for p, s in split.parts.items()} == {"train": 2, "test": 2}

    def test_no_subject_appears_twice(self) -> None:
        """Participant leakage would inflate scores misleadingly."""
        split = make_splits(CHINESEEG, seed=3)
        validate_subject_disjoint(split, CHINESEEG)

    def test_every_subject_is_assigned_exactly_once(self) -> None:
        split = make_splits(CHINESEEG, seed=1)
        assigned = [s for v in split.parts.values() for s in v]
        assert sorted(assigned) == sorted(p.subject_id for p in CHINESEEG)

    def test_sex_is_balanced_across_parts_for_chineseeeg(self) -> None:
        """5F/5M split 3:1:1 should give 3F+3M / 1F+1M / 1F+1M."""
        split = make_splits(CHINESEEG, seed=0)
        by_id = {p.subject_id: p for p in CHINESEEG}
        counts = {
            part: Counter(by_id[s].sex for s in subjects) for part, subjects in split.parts.items()
        }
        assert counts["train"] == Counter({"F": 3, "M": 3})
        assert counts["valid"] == Counter({"F": 1, "M": 1})
        assert counts["test"] == Counter({"F": 1, "M": 1})

    def test_sex_is_balanced_for_chisco(self) -> None:
        """2F/2M split 1:1 must give one of each per part."""
        split = make_splits(CHISCO, ratios={"train": 1, "test": 1}, seed=0)
        by_id = {p.subject_id: p for p in CHISCO}
        for part in ("train", "test"):
            sexes = Counter(by_id[s].sex for s in split.parts[part])
            assert sexes == Counter({"F": 1, "M": 1}), f"{part}: {sexes}"

    def test_balance_is_stable_across_seeds(self) -> None:
        """Balance must not depend on a lucky seed."""
        for seed in range(8):
            split = make_splits(CHINESEEG, seed=seed)
            by_id = {p.subject_id: p for p in CHINESEEG}
            for part in ("valid", "test"):
                sexes = Counter(by_id[s].sex for s in split.parts[part])
                assert sexes["F"] == sexes["M"], f"seed={seed} {part}: {sexes}"

    def test_same_seed_reproduces_split(self) -> None:
        a = make_splits(CHINESEEG, seed=7)
        b = make_splits(CHINESEEG, seed=7)
        assert a.parts == b.parts

    def test_different_seeds_can_differ(self) -> None:
        a = make_splits(CHINESEEG, seed=0)
        b = make_splits(CHINESEEG, seed=5)
        assert a.parts != b.parts

    def test_marks_split_as_subject_level(self) -> None:
        assert make_splits(CHINESEEG).subject_level is True

    def test_rejects_empty_participants(self) -> None:
        with pytest.raises(SplitError, match="no participants"):
            make_splits([])

    def test_rejects_duplicate_subject_ids(self) -> None:
        dupes = [Participant("a", "d", "F"), Participant("a", "d", "M")]
        with pytest.raises(SplitError, match="duplicate subject ids"):
            make_splits(dupes)

    def test_rejects_unknown_level(self) -> None:
        with pytest.raises(SplitError, match="level must be"):
            make_splits(CHINESEEG, level="session")

    def test_rejects_unknown_balance(self) -> None:
        with pytest.raises(SplitError, match="balance must be"):
            make_splits(CHINESEEG, balance="age")


class TestMergedDatasets:
    def test_merged_split_covers_both_datasets(self) -> None:
        merged = [*CHISCO, *CHINESEEG]
        split = make_splits(merged, seed=0)
        assigned = [s for v in split.parts.values() for s in v]
        assert any(s.startswith("chisco:") for s in assigned)
        assert any(s.startswith("chineseeeg:") for s in assigned)

    def test_merged_subjects_are_disjoint(self) -> None:
        merged = [*CHISCO, *CHINESEEG]
        validate_subject_disjoint(make_splits(merged, seed=2), merged)

    def test_namespacing_prevents_collision_between_datasets(self) -> None:
        """Both datasets use sub-NN, so ids must be namespaced or they collide."""
        merged = [*CHISCO, *CHINESEEG]
        ids = [p.subject_id for p in merged]
        assert len(set(ids)) == len(ids)
        # Without namespacing, chisco:sub-04 and chineseeeg:sub-04 would clash.
        assert "chisco:sub-04" in ids and "chineseeeg:sub-04" in ids


class TestRunLevelSplits:
    def test_run_level_is_flagged_not_subject_level(self) -> None:
        runs = {p.subject_id: [f"run-{i:02d}" for i in range(10)] for p in CHINESEEG}
        split = make_splits(
            CHINESEEG,
            ratios={"train": 3, "valid": 1, "test": 1},
            level="run",
            runs_per_subject=runs,
        )
        assert split.subject_level is False

    def test_run_level_emits_a_warning(self) -> None:
        """Users must not mistake a within-participant score for generalisation."""
        runs = {p.subject_id: [f"run-{i:02d}" for i in range(10)] for p in CHINESEEG}
        split = make_splits(CHINESEEG, level="run", runs_per_subject=runs)
        assert split.warnings
        assert "NOT a cross-subject estimate" in split.warnings[0]

    def test_run_level_shares_participants_across_parts(self) -> None:
        runs = {p.subject_id: [f"run-{i:02d}" for i in range(10)] for p in CHINESEEG}
        split = make_splits(CHINESEEG, level="run", runs_per_subject=runs)
        train_subjects = {r.split("/")[0] for r in split.parts["train"]}
        test_subjects = {r.split("/")[0] for r in split.parts["test"]}
        assert train_subjects & test_subjects, "run-level split should overlap subjects"

    def test_run_level_requires_runs(self) -> None:
        with pytest.raises(SplitError, match="requires runs_per_subject"):
            make_splits(CHINESEEG, level="run")

    def test_subject_level_has_no_warning(self) -> None:
        assert make_splits(CHINESEEG).warnings == ()


class TestValidateSubjectDisjoint:
    def test_detects_leakage(self) -> None:
        from bci_eeg_to_text.splits import Split

        leaking = Split(parts={"train": ("a", "b"), "test": ("b", "c")})
        with pytest.raises(SplitError, match="appears in both"):
            validate_subject_disjoint(leaking, ["a", "b", "c"])

    def test_passes_when_disjoint(self) -> None:
        from bci_eeg_to_text.splits import Split

        validate_subject_disjoint(Split(parts={"train": ("a",), "test": ("b",)}), ["a", "b"])


class TestSexBalanceReporting:
    def test_reports_counts_per_part(self) -> None:
        split = make_splits(CHISCO, ratios={"train": 1, "test": 1}, seed=0)
        balance = split.sex_balance({p.subject_id: p for p in CHISCO})
        assert balance["train"] == {"F": 1, "M": 1}
        assert balance["test"] == {"F": 1, "M": 1}

    def test_handles_missing_sex(self) -> None:
        unknown = [Participant("a", "d", None), Participant("b", "d", None)]
        split = make_splits(unknown, ratios={"train": 1, "test": 1})
        assert split.sex_balance({p.subject_id: p for p in unknown})["train"] == {"unknown": 1}
