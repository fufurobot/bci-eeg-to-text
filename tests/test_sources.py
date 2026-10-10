"""Tests for the dataset-source abstraction."""

from __future__ import annotations

from pathlib import Path

import pytest

from bci_eeg_to_text.sources import (
    ExtractionReport,
    Participant,
    RunInfo,
    available_sources,
    get_source,
    load_participants_tsv,
    register_source,
    subjects_for_split,
    usable_subjects,
)


class TestParticipantTsv:
    def test_parses_id_and_sex(self, tmp_path: Path) -> None:
        tsv = tmp_path / "participants.tsv"
        tsv.write_text("participant_id\tage\tsex\nsub-01\t26\tF\nsub-02\t30\tM\n", encoding="utf-8")
        parsed = load_participants_tsv(tsv)
        assert [p.subject_id for p in parsed] == ["sub-01", "sub-02"]
        assert [p.sex for p in parsed] == ["F", "M"]

    def test_parses_chineseeeg_layout_without_age(self, tmp_path: Path) -> None:
        tsv = tmp_path / "participants.tsv"
        tsv.write_text(
            "participant_id\tsex\thand\tweight\theight\nsub-04\tF\tR\tn/a\tn/a\n",
            encoding="utf-8",
        )
        assert load_participants_tsv(tsv)[0].sex == "F"

    def test_handles_utf8_bom(self, tmp_path: Path) -> None:
        """BIDS files are often written with a BOM, which breaks naive parsing."""
        tsv = tmp_path / "participants.tsv"
        tsv.write_bytes("participant_id\tsex\nsub-01\tF\n".encode("utf-8-sig"))
        assert load_participants_tsv(tsv)[0].subject_id == "sub-01"

    def test_missing_sex_becomes_none(self, tmp_path: Path) -> None:
        tsv = tmp_path / "participants.tsv"
        tsv.write_text("participant_id\tsex\nsub-01\tn/a\n", encoding="utf-8")
        assert load_participants_tsv(tsv)[0].sex is None

    def test_absent_file_returns_empty(self, tmp_path: Path) -> None:
        assert load_participants_tsv(tmp_path / "nope.tsv") == []

    def test_missing_id_column_raises(self, tmp_path: Path) -> None:
        tsv = tmp_path / "participants.tsv"
        tsv.write_text("subject\tsex\nsub-01\tF\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing participant_id"):
            load_participants_tsv(tsv)


class TestUsableSubjects:
    def test_drops_participants_without_runs(self) -> None:
        """A listed subject with no local data must not occupy a split slot."""
        people = [Participant("sub-01", sex="F"), Participant("sub-03", sex="M")]
        runs = [RunInfo("sub-01", "d", "run-001", Path("x"))]
        assert [p.subject_id for p in usable_subjects(people, runs)] == ["sub-01"]

    def test_keeps_all_when_all_have_runs(self) -> None:
        people = [Participant("sub-01", sex="F"), Participant("sub-02", sex="M")]
        runs = [
            RunInfo("sub-01", "d", "run-001", Path("x")),
            RunInfo("sub-02", "d", "run-002", Path("y")),
        ]
        assert len(usable_subjects(people, runs)) == 2


class TestRegistry:
    def test_builtin_sources_are_registered(self) -> None:
        names = available_sources()
        assert "chisco" in names
        assert "chineseeeg" in names

    def test_get_source_returns_usable_object(self) -> None:
        source = get_source("chisco")
        assert source.name == "chisco"
        assert isinstance(source.describe(), str)

    def test_unknown_source_raises_with_options(self) -> None:
        with pytest.raises(KeyError, match="unknown dataset"):
            get_source("nope")

    def test_register_requires_a_name(self) -> None:
        class Nameless:
            pass

        with pytest.raises(ValueError, match="must define a non-empty 'name'"):
            register_source(Nameless)


class TestSubjectsForSplit:
    def test_namespaces_ids_across_sources(self) -> None:
        """ds005170 and ds004952 both use sub-NN, so ids must not collide."""
        from bci_eeg_to_text.sources import DatasetSource  # noqa: F401

        class Fake:
            def __init__(self, name: str, ids: list[str]) -> None:
                self.name = name
                self._ids = ids

            def participants(self) -> list[Participant]:
                return [Participant(i, sex="F") for i in self._ids]

        merged = subjects_for_split([Fake("chisco", ["sub-04"]), Fake("chineseeeg", ["sub-04"])])
        ids = [p.subject_id for p in merged]
        assert ids == ["chisco:sub-04", "chineseeeg:sub-04"]
        assert len(set(ids)) == 2

    def test_records_dataset_on_each_participant(self) -> None:
        class Fake:
            name = "d"

            def participants(self) -> list[Participant]:
                return [Participant("sub-01", sex="M")]

        assert subjects_for_split([Fake()])[0].dataset == "d"


class TestExtractionReport:
    def test_ok_when_not_skipped(self) -> None:
        report = ExtractionReport(run=RunInfo("s", "d", "r", Path("p")), n_segments=5)
        assert report.ok

    def test_not_ok_when_skipped(self) -> None:
        report = ExtractionReport(
            run=RunInfo("s", "d", "r", Path("p")), n_segments=0, skipped_reason="placeholder"
        )
        assert not report.ok


class TestRunInfo:
    def test_namespaced_subject(self) -> None:
        assert RunInfo("sub-01", "chisco", "run-001", Path("p")).namespaced_subject == (
            "chisco:sub-01"
        )


@pytest.mark.realdata
class TestRealSources:
    def test_chisco_reports_four_usable_subjects(self) -> None:
        """sub-03 is placeholder-only, so only 4 subjects are usable."""
        source = get_source("chisco")
        if not source.root().exists():
            pytest.skip("dataset not mounted")
        runs = source.runs()
        subjects = sorted({r.subject_id for r in runs})
        assert subjects == ["sub-01", "sub-02", "sub-04", "sub-05"]
        # And participants() must agree, including sex for balanced splitting.
        participants = source.participants()
        assert sorted(p.short_id for p in participants) == subjects
        assert {p.sex for p in participants} == {"F", "M"}

    def test_chisco_run_count_matches_usable_files(self) -> None:
        source = get_source("chisco")
        if not source.root().exists():
            pytest.skip("dataset not mounted")
        # 223 paths minus 46 placeholders (truncated files are still enumerated
        # here and surface as read errors at extraction time).
        assert len(source.runs()) == 177

    def test_chineseeeg_reports_ten_subjects(self) -> None:
        source = get_source("chineseeeg")
        if not source.root().exists():
            pytest.skip("dataset not mounted")
        participants = source.participants()
        assert len(participants) == 10
        assert sum(1 for p in participants if p.sex == "F") == 5
        assert sum(1 for p in participants if p.sex == "M") == 5


class TestChineseEEGAlignment:
    """Guards for the ChineseEEG row-to-text mapping.

    Row boundaries come from ROWS/ROWE markers, and a row's duration is its
    character count times 0.35 s. Both facts are asserted here because getting
    them wrong produces confidently mislabelled training data.
    """

    def test_chapter_banners_are_kept_as_rows(self, tmp_path: Path) -> None:
        """Numeric chapter banners are displayed, so they occupy a row.

        Dropping them desynchronises every later row.
        """
        import zipfile

        from bci_eeg_to_text.chineseeeg import read_segmented_novel

        sheet = (
            "<worksheet><sheetData>"
            '<row r="1"><c t="inlineStr"><is><t>Chinese_text</t></is></c></row>'
            '<row r="2"><c t="inlineStr"><is><t>1</t></is></c></row>'
            '<row r="3"><c t="inlineStr"><is><t>我六岁那年</t></is></c></row>'
            "</sheetData></worksheet>"
        )
        path = tmp_path / "segmented_run_1.xlsx"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("xl/worksheets/sheet1.xml", sheet)
        assert read_segmented_novel(path) == ["1", "我六岁那年"]

    def test_blank_rows_are_dropped(self, tmp_path: Path) -> None:
        import zipfile

        from bci_eeg_to_text.chineseeeg import read_segmented_novel

        sheet = (
            "<worksheet><sheetData>"
            '<row r="1"><c t="inlineStr"><is><t>Chinese_text</t></is></c></row>'
            '<row r="2"></row>'
            '<row r="3"><c t="inlineStr"><is><t>好</t></is></c></row>'
            "</sheetData></worksheet>"
        )
        path = tmp_path / "s.xlsx"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("xl/worksheets/sheet1.xml", sheet)
        assert read_segmented_novel(path) == ["好"]

    def test_correlation_is_high_when_duration_tracks_length(self) -> None:
        from bci_eeg_to_text.chineseeeg import duration_length_correlation

        rate = 256.0
        texts = ["a" * n for n in (2, 4, 6, 8, 10, 12, 14, 16)]
        intervals = []
        cursor = 0
        for text in texts:
            stop = cursor + int(len(text) * 0.35 * rate)
            intervals.append((cursor, stop))
            cursor = stop
        assert duration_length_correlation(intervals, texts, rate) > 0.99

    def test_correlation_is_negative_when_intervals_are_inverted(self) -> None:
        """The signature of the corrupt run: shorter text, longer interval."""
        from bci_eeg_to_text.chineseeeg import duration_length_correlation

        rate = 256.0
        texts = ["a" * n for n in (2, 4, 6, 8, 10, 12, 14, 16)]
        intervals = []
        cursor = 0
        for text in reversed(texts):
            stop = cursor + int(len(text) * 0.35 * rate)
            intervals.append((cursor, stop))
            cursor = stop
        assert duration_length_correlation(intervals, texts, rate) < 0

    def test_correlation_returns_none_for_tiny_inputs(self) -> None:
        from bci_eeg_to_text.chineseeeg import duration_length_correlation

        assert duration_length_correlation([(0, 10)], ["ab"], 256.0) is None

    def test_correlation_returns_none_without_variance(self) -> None:
        from bci_eeg_to_text.chineseeeg import duration_length_correlation

        intervals = [(i * 100, i * 100 + 50) for i in range(10)]
        assert duration_length_correlation(intervals, ["ab"] * 10, 256.0) is None

    def test_row_intervals_pairs_rows_with_the_next_rowe(self) -> None:
        from bci_eeg_to_text.chineseeeg import Event, row_intervals

        events = [
            Event(0.0, 0.0, "ROWS", 0),
            Event(1.0, 0.0, "ROWE", 256),
            Event(1.1, 0.0, "ROWS", 280),
            Event(2.1, 0.0, "ROWE", 540),
        ]
        assert row_intervals(events) == [(0, 256), (280, 540)]

    def test_row_intervals_raises_on_unbalanced_markers(self) -> None:
        """An unclosed ROWS would shift every later row's text."""
        from bci_eeg_to_text.chineseeeg import Event, row_intervals

        events = [Event(0.0, 0.0, "ROWS", 0), Event(1.0, 0.0, "ROWE", 256)]
        events.append(Event(2.0, 0.0, "ROWS", 600))
        with pytest.raises(Exception, match="no following ROWE|unbalanced"):
            row_intervals(events)


@pytest.mark.realdata
class TestChineseEEGRealAlignment:
    def test_healthy_runs_match_exactly_with_high_correlation(self) -> None:
        """Runs whose counts agree must also show duration proportional to length."""
        from bci_eeg_to_text.chineseeeg import (
            MIN_DURATION_LENGTH_CORRELATION,
            duration_length_correlation,
            read_events,
            read_segmented_novel,
            row_intervals,
        )
        from bci_eeg_to_text.sources import get_source

        source = get_source("chineseeeg")
        if not source.root().exists():
            pytest.skip("dataset not mounted")
        runs = [
            r for r in source.runs() if r.subject_id == "sub-04" and r.run in {"run-02", "run-03"}
        ]
        assert runs, "expected sub-04 runs to exist"
        for run in runs:
            paths = [p for p in source.novel_paths(run) if "_display" not in p.name]
            if not paths:
                continue
            texts = read_segmented_novel(paths[0])
            intervals = row_intervals(read_events(source.events_path(run)))
            assert len(intervals) == len(texts)
            corr = duration_length_correlation(intervals, texts, 256.0)
            assert corr >= MIN_DURATION_LENGTH_CORRELATION, f"{run.run}: corr={corr}"


class TestVhdrRepair:
    """The GarnettDream sidecars reference filenames that do not exist.

    Six of the dataset's sessions ship headers pointing at ``..._GranettDream_``
    while the files on disk are ``..._GarnettDream_`` (one "t"). BrainVision
    resolves those paths literally, so MNE cannot open 126 of 168 headers.
    """

    def test_fix_corrects_the_typo(self) -> None:
        from bci_eeg_to_text.chineseeeg import fix_vhdr_reference

        text = "DataFile=sub-04_ses-GranettDream_task-reading_run-01_eeg.eeg\n"
        assert fix_vhdr_reference(text) == (
            "DataFile=sub-04_ses-GarnettDream_task-reading_run-01_eeg.eeg\n"
        )

    def test_fix_leaves_correct_text_untouched(self) -> None:
        from bci_eeg_to_text.chineseeeg import fix_vhdr_reference

        text = "DataFile=sub-04_ses-GarnettDream_task-reading_run-01_eeg.eeg\n"
        assert fix_vhdr_reference(text) == text

    def test_detects_missing_reference(self, tmp_path: Path) -> None:
        from bci_eeg_to_text.chineseeeg import references_missing_files

        vhdr = tmp_path / "h.vhdr"
        vhdr.write_text("x")
        assert references_missing_files(vhdr, "DataFile=absent.eeg\n")

    def test_present_reference_is_reported_ok(self, tmp_path: Path) -> None:
        from bci_eeg_to_text.chineseeeg import references_missing_files

        (tmp_path / "present.eeg").write_bytes(b"x")
        vhdr = tmp_path / "h.vhdr"
        vhdr.write_text("x")
        assert not references_missing_files(vhdr, "DataFile=present.eeg\n")

    def test_readable_vhdr_repairs_via_mirror(self, tmp_path: Path) -> None:
        """A read-only dataset must still be readable through a mirror."""
        from bci_eeg_to_text.chineseeeg import readable_vhdr

        data = tmp_path / "sub-04" / "ses-GarnettDream" / "eeg"
        data.mkdir(parents=True)
        (data / "sub-04_ses-GarnettDream_task-reading_run-01_eeg.eeg").write_bytes(b"data")
        (data / "sub-04_ses-GarnettDream_task-reading_run-01_eeg.vmrk").write_text("m")
        vhdr = data / "sub-04_ses-GarnettDream_task-reading_run-01_eeg.vhdr"
        vhdr.write_text(
            "DataFile=sub-04_ses-GranettDream_task-reading_run-01_eeg.eeg\n"
            "MarkerFile=sub-04_ses-GranettDream_task-reading_run-01_eeg.vmrk\n"
        )

        repaired, changed = readable_vhdr(vhdr, tmp_path / "work")
        assert changed is True
        assert repaired != vhdr
        text = repaired.read_text()
        assert "GranettDream" not in text
        # The mirror must actually resolve to the real data.
        assert (repaired.parent / "sub-04_ses-GarnettDream_task-reading_run-01_eeg.eeg").exists()

    def test_readable_vhdr_is_a_noop_when_fine(self, tmp_path: Path) -> None:
        from bci_eeg_to_text.chineseeeg import readable_vhdr

        (tmp_path / "a.eeg").write_bytes(b"x")
        vhdr = tmp_path / "a.vhdr"
        vhdr.write_text("DataFile=a.eeg\n")
        path, changed = readable_vhdr(vhdr, tmp_path / "work")
        assert (path, changed) == (vhdr, False)

    def test_readable_vhdr_gives_up_when_unfixable(self, tmp_path: Path) -> None:
        """If the corrected name is also absent, return the original unchanged."""
        from bci_eeg_to_text.chineseeeg import readable_vhdr

        vhdr = tmp_path / "a.vhdr"
        vhdr.write_text("DataFile=sub-01_ses-GranettDream_missing.eeg\n")
        path, changed = readable_vhdr(vhdr, tmp_path / "work")
        assert (path, changed) == (vhdr, False)
