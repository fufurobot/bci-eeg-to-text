"""Tests for the leakage matrix: subject, text, sex and run axes.

The design goal is that no axis is ever *silent*. "No output" must never be
readable as "no leakage", so an unenforced axis has to say so, and a violated
axis has to name what leaks.
"""

from __future__ import annotations

import pytest

from bci_eeg_to_text.leakage import (
    DEFAULT_LEAKAGE_MODE,
    DEFAULT_SUBJECT_MODE,
    LEAKAGE_MODES,
    SUBJECT_MODES,
    LeakageError,
    TextUnit,
    assign_units_to_parts,
    check_run_axis,
    check_sex_axis,
    check_subject_axis,
    check_text_axis,
    describe_subject_mode,
    describe_text_mode,
    evaluate_matrix,
    group_ids,
    pick_one_subject,
    summarize_coverage,
    supports_text_mode,
    units_from_labels,
)
from bci_eeg_to_text.splits import Participant
from bci_eeg_to_text.textstructure import RowLabel


def units(spec: list[tuple[str, str, str]]) -> list[TextUnit]:
    """Build units from ``(row_id, paragraph_id, article_id)`` triples."""
    return [TextUnit(row_id=r, text=f"text-{r}", paragraph_id=p, article_id=a) for r, p, a in spec]


TWO_ARTICLES = units(
    [
        ("r0", "p0", "Ch0"),
        ("r1", "p0", "Ch0"),
        ("r2", "p1", "Ch0"),
        ("r3", "p1", "Ch0"),
        ("r4", "p2", "Ch1"),
        ("r5", "p2", "Ch1"),
        ("r6", "p3", "Ch1"),
        ("r7", "p3", "Ch1"),
    ]
)


class TestSubjectAxis:
    def test_cross_subject_accepts_disjoint_parts(self) -> None:
        report = check_subject_axis({"train": ["a"], "test": ["b"]}, "cross_subject")
        assert report.ok and report.enforced

    def test_cross_subject_detects_a_shared_participant(self) -> None:
        report = check_subject_axis({"train": ["a"], "test": ["a"]}, "cross_subject")
        assert not report.ok
        assert "a" in report.violations[0]

    def test_one_subject_accepts_a_single_participant(self) -> None:
        assert check_subject_axis({"train": ["a"], "test": ["a"]}, "one_subject").ok

    def test_one_subject_rejects_several(self) -> None:
        report = check_subject_axis({"train": ["a", "b"]}, "one_subject")
        assert not report.ok
        assert "exactly 1" in report.violations[0]

    def test_none_is_unenforced_but_still_reports(self) -> None:
        report = check_subject_axis({"train": ["a"], "test": ["a"]}, "none")
        assert not report.enforced
        assert report.ok  # nothing is claimed
        assert "NOT ENFORCED" in report.describe()

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(LeakageError, match="unknown subject mode"):
            check_subject_axis({"train": ["a"]}, "cross_participant")


class TestTextAxis:
    def test_row_mode_ignores_shared_paragraphs(self) -> None:
        parts = {
            "train": units([("r0", "p0", "Ch0")]),
            "test": units([("r1", "p0", "Ch0")]),
        }
        assert check_text_axis(parts, "row").ok

    def test_paragraph_mode_detects_a_shared_paragraph(self) -> None:
        parts = {
            "train": units([("r0", "p0", "Ch0")]),
            "test": units([("r1", "p0", "Ch0")]),
        }
        report = check_text_axis(parts, "paragraph")
        assert not report.ok and "p0" in report.violations[0]

    def test_article_mode_detects_rows_of_one_chapter(self) -> None:
        parts = {
            "train": units([("r0", "p0", "Ch0")]),
            "test": units([("r1", "p1", "Ch0")]),
        }
        assert not check_text_axis(parts, "article").ok

    def test_none_is_unenforced(self) -> None:
        report = check_text_axis({"train": TWO_ARTICLES}, "none")
        assert not report.enforced
        assert "NOT ENFORCED" in report.describe()

    def test_unplaced_row_falls_back_to_its_own_id(self) -> None:
        orphan = units([("r0", None, None), ("r1", None, None)])
        assert group_ids(orphan, "paragraph") == {"r0", "r1"}

    def test_group_ids_by_mode(self) -> None:
        assert group_ids(TWO_ARTICLES, "paragraph") == {"p0", "p1", "p2", "p3"}
        assert group_ids(TWO_ARTICLES, "article") == {"Ch0", "Ch1"}


class TestSexAxis:
    def test_balanced_parts_pass(self) -> None:
        people = {"a": Participant("a", sex="F"), "b": Participant("b", sex="M")}
        assert check_sex_axis({"train": ["a", "b"]}, people).ok

    def test_imbalance_is_reported_with_counts(self) -> None:
        people = {"a": Participant("a", sex="F"), "b": Participant("b", sex="F")}
        report = check_sex_axis({"train": ["a", "b"]}, people)
        assert not report.ok
        assert "|F-M|=2" in report.violations[0]

    def test_single_subject_part_is_imbalanced(self) -> None:
        """|F-M| = 1 for any single participant, by construction."""
        report = check_sex_axis({"test": ["a"]}, {"a": Participant("a", sex="F")})
        assert not report.ok

    def test_unknown_sex_is_reported_not_assumed_balanced(self) -> None:
        report = check_sex_axis({"train": ["a"]}, {"a": Participant("a", sex=None)})
        assert not report.ok
        assert any("no recorded sex" in v for v in report.violations)

    def test_counts_are_included_in_detail(self) -> None:
        people = {"a": Participant("a", sex="F"), "b": Participant("b", sex="M")}
        report = check_sex_axis({"train": ["a", "b"]}, people)
        assert report.detail["train"] == {"F": 1, "M": 1}


class TestRunAxis:
    def test_disjoint_runs_pass(self) -> None:
        assert check_run_axis({"train": ["r1"], "test": ["r2"]}).ok

    def test_a_shared_recording_is_detected(self) -> None:
        """This is the bug the matrix caught: one run in two parts."""
        report = check_run_axis({"train": ["r1"], "test": ["r1"]})
        assert not report.ok
        assert "r1" in report.violations[0]


class TestMatrix:
    def _matrix(self, **kwargs):
        # Two subjects per part so each part is sex-balanced: a single-subject
        # part is imbalanced by construction (|F-M| = 1), which the sex axis
        # reports -- correctly, and it would otherwise make this "clean" case
        # fail for a reason unrelated to what it tests.
        defaults = dict(
            subject_parts={"train": ["a", "b"], "test": ["c", "d"]},
            participants={
                "a": Participant("a", sex="F"),
                "b": Participant("b", sex="M"),
                "c": Participant("c", sex="F"),
                "d": Participant("d", sex="M"),
            },
            subject_mode="cross_subject",
            text_mode="row",
        )
        defaults.update(kwargs)
        return evaluate_matrix(**defaults)

    def test_clean_split_reports_no_leakage(self) -> None:
        assert self._matrix().ok

    def test_lists_unenforced_axes(self) -> None:
        matrix = self._matrix(subject_mode="none", text_mode="none")
        assert set(matrix.unenforced_axes) == {"subject", "text"}
        assert "upper bound" in matrix.describe()

    def test_names_exactly_which_axis_leaks(self) -> None:
        matrix = self._matrix(
            text_parts={
                "train": units([("r0", "p0", "Ch0")]),
                "test": units([("r1", "p0", "Ch0")]),
            },
            text_mode="paragraph",
        )
        assert "text" in matrix.leaking_axes
        assert "subject" not in matrix.leaking_axes

    def test_describe_is_never_silent(self) -> None:
        """Even a fully-clean split must say so explicitly."""
        text = self._matrix().describe()
        assert "leakage matrix" in text
        assert "subject=" in text and "text=" in text
        assert "sex=" in text and "run=" in text

    def test_describe_names_what_leaks(self) -> None:
        matrix = self._matrix(
            subject_parts={"train": ["a", "b"], "test": ["a", "d"]},
            subject_mode="cross_subject",
        )
        text = matrix.describe()
        assert "LEAKING" in text
        assert "subject" in text

    def test_unenforced_axis_says_upper_bound(self) -> None:
        text = self._matrix(subject_mode="none").describe()
        assert "NOT ENFORCED" in text
        assert "upper bound" in text

    def test_sex_can_be_skipped_for_single_subject_splits(self) -> None:
        matrix = self._matrix(check_sex=False)
        assert not matrix.sex.enforced

    def test_all_axes_present(self) -> None:
        assert len(self._matrix().axes) == 4


class TestAssignUnits:
    @pytest.mark.parametrize("mode", LEAKAGE_MODES)
    def test_every_row_assigned_once(self, mode: str) -> None:
        parts = assign_units_to_parts(TWO_ARTICLES, mode=mode, ratios={"train": 1, "test": 1})
        assigned = [u.row_id for group in parts.values() for u in group]
        assert sorted(assigned) == sorted(u.row_id for u in TWO_ARTICLES)

    @pytest.mark.parametrize("mode", ["row", "paragraph", "article"])
    def test_assignment_is_leakage_free(self, mode: str) -> None:
        parts = assign_units_to_parts(TWO_ARTICLES, mode=mode, ratios={"train": 1, "test": 1})
        assert check_text_axis(parts, mode).ok

    def test_groups_are_indivisible(self) -> None:
        """Half a paragraph in train and half in test is the leak to avoid."""
        parts = assign_units_to_parts(
            TWO_ARTICLES, mode="paragraph", ratios={"train": 1, "test": 1}
        )
        for group in parts.values():
            for paragraph in {u.paragraph_id for u in group}:
                owners = [
                    part
                    for part, units_ in parts.items()
                    if any(u.paragraph_id == paragraph for u in units_)
                ]
                assert len(owners) == 1

    def test_empty_input(self) -> None:
        assert assign_units_to_parts([], mode="row", ratios={"train": 1}) == {"train": ()}


class TestPickOneSubject:
    def test_picks_exactly_one(self) -> None:
        people = [
            Participant("chineseeeg:a", "chineseeeg", "F"),
            Participant("chineseeeg:b", "chineseeeg", "M"),
        ]
        assert pick_one_subject(people, seed=0) in people

    def test_is_deterministic_for_a_seed(self) -> None:
        people = [Participant(f"chineseeeg:s{i}", "chineseeeg", "F") for i in range(5)]
        assert pick_one_subject(people, seed=3) == pick_one_subject(people, seed=3)

    def test_ignores_sex_balance(self) -> None:
        """With one subject there is nothing to balance."""
        people = [Participant("chineseeeg:a", "chineseeeg", "F")]
        assert pick_one_subject(people).sex == "F"

    def test_paragraph_mode_restricts_to_document_datasets(self) -> None:
        """Only ChineseEEG ships source documents, so only it is eligible."""
        people = [
            Participant("chisco:a", "chisco", "F"),
            Participant("chineseeeg:b", "chineseeeg", "M"),
        ]
        for seed in range(10):
            assert (
                pick_one_subject(people, text_mode="paragraph", seed=seed).dataset == "chineseeeg"
            )

    def test_row_mode_allows_any_dataset(self) -> None:
        people = [Participant("chisco:a", "chisco", "F")]
        assert pick_one_subject(people, text_mode="row").dataset == "chisco"

    def test_raises_when_no_dataset_supports_the_mode(self) -> None:
        people = [Participant("chisco:a", "chisco", "F")]
        with pytest.raises(LeakageError, match="source documents"):
            pick_one_subject(people, text_mode="paragraph")


class TestHelpers:
    def test_defaults(self) -> None:
        assert DEFAULT_SUBJECT_MODE == "cross_subject"
        assert DEFAULT_LEAKAGE_MODE == "row"

    def test_supports_text_mode(self) -> None:
        assert supports_text_mode("chisco", "row")
        assert not supports_text_mode("chisco", "paragraph")
        assert supports_text_mode("chineseeeg", "article")

    @pytest.mark.parametrize("mode", SUBJECT_MODES)
    def test_subject_modes_are_described(self, mode: str) -> None:
        assert "unknown" not in describe_subject_mode(mode)

    @pytest.mark.parametrize("mode", LEAKAGE_MODES)
    def test_text_modes_are_described(self, mode: str) -> None:
        assert "unknown" not in describe_text_mode(mode)

    def test_coverage_reports_fraction(self) -> None:
        mixed = units([("r0", "p0", "Ch0"), ("r1", None, None)])
        assert "50.0%" in summarize_coverage(mixed, "paragraph")

    def test_units_from_labels(self) -> None:
        converted = units_from_labels([RowLabel("x", "d#r0", "d#p0", "Ch1")], dataset="chineseeeg")
        assert converted[0].dataset == "chineseeeg"
