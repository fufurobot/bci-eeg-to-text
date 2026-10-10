"""Tests for text structure: rows, paragraphs and articles."""

from __future__ import annotations

from pathlib import Path

import pytest

from bci_eeg_to_text.textstructure import (
    DEFAULT_LEAKAGE_MODE,
    LEAKAGE_MODES,
    TextStructureError,
    label_rows,
    parse_document,
    read_document,
)

SAMPLE = """Ch0

献给莱翁·维尔特，

请孩子们原谅我把这本书献给了一个大人。
我有一个很认真的理由。

Ch1

我六岁那年，在一本描写原始森林的
名叫《真实的故事》的书上，

看见过一幅精彩的插图。
"""


class TestParseDocument:
    def test_finds_chapters(self) -> None:
        doc = parse_document(SAMPLE, "novel")
        assert [a[0] for a in doc.articles] == ["Ch0", "Ch1"]

    def test_finds_paragraphs(self) -> None:
        """Paragraphs are runs of consecutive non-blank lines.

        The sample has four: the dedication, the two-line author's note (which
        is ONE paragraph despite spanning two lines), and two more after Ch1.
        """
        doc = parse_document(SAMPLE, "novel")
        assert len(doc.paragraphs) == 4

    def test_multiline_paragraph_is_not_split(self) -> None:
        """A paragraph spans consecutive lines, so line breaks do not divide."""
        doc = parse_document(SAMPLE, "novel")
        # The Ch1 paragraph spans two source lines.
        assert doc.flat[doc.paragraphs[2][0] : doc.paragraphs[2][1]].count("名叫") == 1
        assert "看见过" not in doc.flat[doc.paragraphs[2][0] : doc.paragraphs[2][1]]

    def test_chapter_marker_is_not_prose(self) -> None:
        doc = parse_document(SAMPLE, "novel")
        assert "Ch0" not in doc.flat

    def test_paragraphs_are_ordered_and_disjoint(self) -> None:
        doc = parse_document(SAMPLE, "novel")
        for (_, prev_stop), (next_start, _) in zip(
            doc.paragraphs, doc.paragraphs[1:], strict=False
        ):
            assert prev_stop <= next_start

    def test_paragraph_maps_to_an_article(self) -> None:
        doc = parse_document(SAMPLE, "novel")
        assert doc.paragraph_article[0] == "Ch0"
        assert doc.paragraph_article[-1] == "Ch1"

    def test_blank_only_document_has_no_paragraphs(self) -> None:
        doc = parse_document("\n\n\n", "empty")
        assert doc.paragraphs == []

    def test_whitespace_is_removed_from_offsets(self) -> None:
        """Offsets index the flattened text, so normalisation must be applied."""
        doc = parse_document("你好 世界\n", "d")
        assert " " not in doc.flat


class TestLocate:
    def test_finds_text_and_offset(self) -> None:
        doc = parse_document(SAMPLE, "novel")
        offset = doc.locate("我六岁那年")
        assert offset is not None
        assert doc.flat[offset:].startswith("我六岁那年")

    def test_returns_none_for_absent_text(self) -> None:
        doc = parse_document(SAMPLE, "novel")
        assert doc.locate("这段文字不在原文里") is None

    def test_repeated_text_resolves_to_successive_occurrences(self) -> None:
        """Otherwise every repeat collapses onto the first paragraph."""
        doc = parse_document("重复的句子。\n\n重复的句子。\n", "d")
        first = doc.locate("重复的句子。")
        second = doc.locate("重复的句子。")
        assert first is not None and second is not None and first != second

    def test_normalises_before_searching(self) -> None:
        doc = parse_document("你好，世界。\n", "d")
        assert doc.locate("你好,世界.") is not None


class TestRowLabeling:
    def test_assigns_row_paragraph_and_article(self) -> None:
        doc = parse_document(SAMPLE, "novel")
        labels = label_rows(["我六岁那年", "请孩子们原谅我"], doc)
        assert labels[0].row_id == "novel#r0"
        assert labels[0].article_id == "Ch1"
        assert labels[0].paragraph_id is not None
        assert labels[1].article_id == "Ch0"

    def test_unplaced_row_keeps_its_id_but_no_structure(self) -> None:
        """An unmatched row must not accidentally match a stricter check."""
        doc = parse_document(SAMPLE, "novel")
        labels = label_rows(["完全不在原文中"], doc)
        assert labels[0].row_id == "novel#r0"
        assert labels[0].paragraph_id is None
        assert labels[0].article_id is None

    def test_rows_in_the_same_paragraph_share_a_paragraph_id(self) -> None:
        doc = parse_document(SAMPLE, "novel")
        labels = label_rows(["我六岁那年", "名叫《真实的故事》"], doc)
        assert labels[0].paragraph_id == labels[1].paragraph_id


class TestReadDocument:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(TextStructureError, match="no such text file"):
            read_document(tmp_path / "nope.txt")

    def test_reads_utf8(self, tmp_path: Path) -> None:
        path = tmp_path / "novel.txt"
        path.write_text(SAMPLE, encoding="utf-8")
        assert read_document(path, "novel").articles

    def test_reads_gbk(self, tmp_path: Path) -> None:
        """GarnettDream is stored in GBK, which is why encoding is explicit."""
        path = tmp_path / "other.txt"
        path.write_bytes("Ch0\n\n你好世界。\n".encode("gbk"))
        doc = read_document(path, "GarnettDream")
        assert doc.articles[0][0] == "Ch0"
        assert doc.locate("你好世界") is not None

    def test_survives_undecodable_bytes(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.txt"
        path.write_bytes(b"Ch0\n\n\xca\xfe\xff garbage\n")
        assert read_document(path, "broken").articles


class TestModes:
    def test_modes_are_ordered_weakest_to_strongest(self) -> None:
        assert LEAKAGE_MODES == ("none", "row", "paragraph", "article")

    def test_default_is_row_level(self) -> None:
        """Rows may not repeat across parts, but prose may."""
        assert DEFAULT_LEAKAGE_MODE == "row"


@pytest.mark.realdata
class TestRealNovels:
    def test_parses_both_novels(self) -> None:
        root = Path("eeg-text-bci-data/bci-eeg/ds004952/derivatives/novels/original_novel")
        if not root.exists():
            pytest.skip("dataset not mounted")
        for name in ("LittlePrince", "GarnettDream"):
            doc = read_document(root / f"{name}.txt", name)
            assert doc.paragraphs, f"{name} produced no paragraphs"
            assert doc.articles, f"{name} produced no articles"

    def test_rows_map_to_the_source_text(self) -> None:
        """Rows are verbatim substrings, which is what makes this possible."""
        base = Path("eeg-text-bci-data/bci-eeg/ds004952/derivatives/novels")
        if not base.exists():
            pytest.skip("dataset not mounted")
        from bci_eeg_to_text.chineseeeg import read_segmented_novel

        doc = read_document(base / "original_novel" / "LittlePrince.txt", "LittlePrince")
        segment_dir = base / "segmented_novel" / "LittlePrince"
        run_files = sorted(
            p for p in segment_dir.glob("*.xlsx") if "_display" not in p.name and "run" in p.name
        )
        if not run_files:
            pytest.skip("no segmented runs")
        rows = read_segmented_novel(run_files[0])
        labels = label_rows(rows, doc)
        placed = sum(1 for label in labels if label.paragraph_id)
        assert placed / len(labels) > 0.95, f"only {placed}/{len(labels)} rows mapped"
