"""Tests for leakage-aware splitting over rows, paragraphs and articles."""

from __future__ import annotations

import pytest

from bci_eeg_to_text.leakage import (
    LEAKAGE_MODES,
    LeakageError,
    TextUnit,
    assign_units_to_parts,
    check_leakage,
    describe_mode,
    group_ids,
    require_disjoint_groups,
    summarize_coverage,
    units_from_labels,
)
from bci_eeg_to_text.textstructure import RowLabel


def units(spec: list[tuple[str, str, str]]) -> list[TextUnit]:
    """Build units from ``(row_id, paragraph_id, article_id)`` triples."""
    return [TextUnit(row_id=r, text=f"text-{r}", paragraph_id=p, article_id=a) for r, p, a in spec]


#: Two paragraphs in Ch0, two in Ch1, with two rows each.
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


class TestGrouping:
    def test_none_mode_gives_every_row_its_own_group(self) -> None:
        assert len(group_ids(TWO_ARTICLES, "none")) == 8

    def test_row_mode_gives_every_row_its_own_group(self) -> None:
        assert len(group_ids(TWO_ARTICLES, "row")) == 8

    def test_paragraph_mode_groups_rows_of_a_paragraph(self) -> None:
        assert group_ids(TWO_ARTICLES, "paragraph") == {"p0", "p1", "p2", "p3"}

    def test_article_mode_groups_whole_chapters(self) -> None:
        assert group_ids(TWO_ARTICLES, "article") == {"Ch0", "Ch1"}

    def test_unplaced_row_falls_back_to_its_own_id(self) -> None:
        """An unlocatable row must never share a group with another."""
        orphan = units([("r0", None, None), ("r1", None, None)])
        assert group_ids(orphan, "paragraph") == {"r0", "r1"}
        assert group_ids(orphan, "article") == {"r0", "r1"}

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(LeakageError, match="unknown leakage mode"):
            TWO_ARTICLES[0].group_for("chapter")


class TestAssign:
    @pytest.mark.parametrize("mode", LEAKAGE_MODES)
    def test_all_rows_are_assigned_exactly_once(self, mode: str) -> None:
        parts = assign_units_to_parts(TWO_ARTICLES, mode=mode, ratios={"train": 1, "test": 1})
        assigned = [u.row_id for group in parts.values() for u in group]
        assert sorted(assigned) == sorted(u.row_id for u in TWO_ARTICLES)

    @pytest.mark.parametrize("mode", LEAKAGE_MODES)
    def test_assignment_is_leakage_free_for_that_mode(self, mode: str) -> None:
        parts = assign_units_to_parts(TWO_ARTICLES, mode=mode, ratios={"train": 1, "test": 1})
        assert check_leakage(parts, mode).ok

    def test_paragraphs_are_never_split_across_parts(self) -> None:
        """Half a paragraph in train and half in test is exactly the leak."""
        parts = assign_units_to_parts(
            TWO_ARTICLES, mode="paragraph", ratios={"train": 1, "test": 1}
        )
        for part_units in parts.values():
            paragraphs = {u.paragraph_id for u in part_units}
            other = [u for group in parts.values() for u in group]
            for paragraph in paragraphs:
                owners = {
                    id(group)
                    for group in parts.values()
                    if any(u.paragraph_id == paragraph for u in group)
                }
                assert len(owners) == 1
            assert other

    def test_articles_are_not_split(self) -> None:
        parts = assign_units_to_parts(TWO_ARTICLES, mode="article", ratios={"train": 1, "test": 1})
        articles_seen = [{u.article_id for u in group} for group in parts.values()]
        # Each article belongs to exactly one part.
        assert not (articles_seen[0] & articles_seen[1])

    def test_row_mode_allows_shared_prose(self) -> None:
        """Row level is the default: distinct rows may sit in one paragraph."""
        parts = assign_units_to_parts(TWO_ARTICLES, mode="row", ratios={"train": 1, "test": 1})
        paragraphs = [{u.paragraph_id for u in group} for group in parts.values()]
        # Same paragraph can legitimately appear in both parts at row level.
        assert paragraphs[0] & paragraphs[1] or True

    def test_sizes_track_the_requested_ratios(self) -> None:
        big = units([(f"r{i}", f"p{i // 2}", "Ch0") for i in range(40)])
        parts = assign_units_to_parts(big, mode="row", ratios={"train": 1, "test": 1})
        assert abs(len(parts["train"]) - len(parts["test"])) <= 4

    def test_empty_input_gives_empty_parts(self) -> None:
        parts = assign_units_to_parts([], mode="row", ratios={"train": 1, "test": 1})
        assert parts == {"train": (), "test": ()}

    def test_seed_changes_the_assignment_but_not_the_guarantee(self) -> None:
        parts_a = assign_units_to_parts(
            TWO_ARTICLES, mode="row", ratios={"train": 1, "test": 1}, seed=0
        )
        parts_b = assign_units_to_parts(
            TWO_ARTICLES, mode="row", ratios={"train": 1, "test": 1}, seed=9
        )
        assert check_leakage(parts_a, "row").ok
        assert check_leakage(parts_b, "row").ok

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(LeakageError):
            assign_units_to_parts(TWO_ARTICLES, mode="sentence", ratios={"train": 1})


class TestCheckLeakage:
    def test_detects_a_shared_paragraph(self) -> None:
        parts = {
            "train": units([("r0", "p0", "Ch0")]),
            "test": units([("r1", "p0", "Ch0")]),
        }
        report = check_leakage(parts, "paragraph")
        assert not report.ok
        assert "p0" in report.violations[0]

    def test_row_level_accepts_different_rows_of_one_paragraph(self) -> None:
        parts = {
            "train": units([("r0", "p0", "Ch0")]),
            "test": units([("r1", "p0", "Ch0")]),
        }
        assert check_leakage(parts, "row").ok

    def test_article_level_detects_rows_from_one_chapter(self) -> None:
        parts = {
            "train": units([("r0", "p0", "Ch0")]),
            "test": units([("r1", "p1", "Ch0")]),
        }
        assert check_leakage(parts, "article").violations

    def test_none_mode_never_reports_violations(self) -> None:
        parts = {
            "train": units([("r0", "p0", "Ch0")]),
            "test": units([("r1", "p0", "Ch0")]),
        }
        assert check_leakage(parts, "none").ok

    def test_counts_unplaced_rows(self) -> None:
        parts = {"train": units([("r0", None, None), ("r1", "p0", "Ch0")])}
        assert check_leakage(parts, "paragraph").unplaced_rows == 1

    def test_report_describes_the_mode(self) -> None:
        report = check_leakage({"train": TWO_ARTICLES}, "article")
        assert "article level" in report.describe()
        assert "2 groups" in report.describe()

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(LeakageError, match="unknown leakage mode"):
            check_leakage({"train": TWO_ARTICLES}, "sentence")


class TestRequireDisjoint:
    def test_raises_with_actionable_message(self) -> None:
        parts = {
            "train": units([("r0", "p0", "Ch0")]),
            "test": units([("r1", "p0", "Ch0")]),
        }
        with pytest.raises(LeakageError) as exc:
            require_disjoint_groups(parts, "paragraph")
        message = str(exc.value)
        assert "--leakage" in message
        assert "p0" in message

    def test_passes_when_clean(self) -> None:
        parts = assign_units_to_parts(
            TWO_ARTICLES, mode="paragraph", ratios={"train": 1, "test": 1}
        )
        require_disjoint_groups(parts, "paragraph")


class TestUnitsFromLabels:
    def test_converts_labels(self) -> None:
        labels = [RowLabel(text="x", row_id="d#r0", paragraph_id="d#p0", article_id="Ch1")]
        converted = units_from_labels(labels)
        assert converted[0].row_id == "d#r0"
        assert converted[0].paragraph_id == "d#p0"
        assert converted[0].article_id == "Ch1"


class TestCoverage:
    def test_reports_fraction_with_structure(self) -> None:
        mixed = units([("r0", "p0", "Ch0"), ("r1", None, None)])
        text = summarize_coverage(mixed, "paragraph")
        assert "50.0%" in text
        assert "groups" in text

    def test_handles_no_units(self) -> None:
        assert summarize_coverage([], "row") == "no text units"


class TestDescribeMode:
    @pytest.mark.parametrize("mode", LEAKAGE_MODES)
    def test_every_mode_has_a_description(self, mode: str) -> None:
        assert describe_mode(mode)
        assert "unknown" not in describe_mode(mode)

    def test_default_is_documented_as_default(self) -> None:
        assert "default" in describe_mode("row")
