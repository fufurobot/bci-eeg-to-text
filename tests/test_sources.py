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
