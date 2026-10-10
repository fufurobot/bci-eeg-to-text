"""Tests for reading materialised split trees and building training data.

The central guarantees:

* both subject-level and run-level trees load identically, so the trainer needs
  no special cases, and
* no recording may appear in two parts, because that leaks a subject across
  train and test and inflates scores while looking like a good model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bci_eeg_to_text.splittree import (
    SplitTreeError,
    check_no_leakage,
    iter_part_files,
    load_split_tree,
    summarise,
)


def link_subject(
    split_root: Path, part: str, subject: str, corpus: Path, dataset: str | None = None
):
    """Create a subject-level link, as `split --link` would."""
    parent = split_root / part / dataset if dataset else split_root / part
    parent.mkdir(parents=True, exist_ok=True)
    source = corpus / subject
    source.mkdir(parents=True, exist_ok=True)
    eeg = source / "ses-01" / "eeg"
    eeg.mkdir(parents=True, exist_ok=True)
    (eeg / f"{subject}_ses-01_task-imagine_run-01_eeg.edf").write_bytes(b"0" * 16)
    import os

    link = parent / subject
    link.symlink_to(os.path.relpath(source.resolve(), link.parent.resolve()))


def link_run(
    split_root: Path,
    part: str,
    subject: str,
    session: str,
    run: str,
    corpus: Path,
    dataset: str | None = None,
):
    """Create a run-level link nested by subject and session-qualified run."""
    parent = split_root / part / dataset if dataset else split_root / part
    target_dir = parent / subject / session / run
    target_dir.mkdir(parents=True, exist_ok=True)
    src_dir = corpus / subject / session / "eeg"
    src_dir.mkdir(parents=True, exist_ok=True)
    name = f"{subject}_{session}_task-imagine_{run}_eeg.edf"
    src = src_dir / name
    src.write_bytes(b"0" * 16)
    import os

    (target_dir / name).symlink_to(os.path.relpath(src.resolve(), target_dir.resolve()))
    return src


def write_manifest(split_root: Path, *, subject_level: bool, warnings: list[str] | None = None):
    split_root.mkdir(parents=True, exist_ok=True)
    (split_root / "split.json").write_text(
        json.dumps(
            {
                "subject_level": subject_level,
                "warnings": warnings or [],
                "parts": {"train": ["sub-01"], "test": ["sub-02"]},
            }
        )
    )


class TestLoadSplitTree:
    def test_loads_subject_level_tree(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        root = tmp_path / "splits"
        link_subject(root, "train", "sub-01", corpus)
        link_subject(root, "test", "sub-02", corpus)
        write_manifest(root, subject_level=True)

        tree = load_split_tree(root)
        assert tree.subject_level is True
        assert tree.subjects("train") == ["sub-01"]
        assert tree.subjects("test") == ["sub-02"]

    def test_loads_run_level_tree(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        root = tmp_path / "splits"
        link_run(root, "train", "sub-01", "ses-01", "run-01", corpus)
        link_run(root, "test", "sub-01", "ses-01", "run-02", corpus)
        write_manifest(root, subject_level=False)

        tree = load_split_tree(root)
        assert tree.subject_level is False
        # Both parts share the participant, which is what run level means.
        assert tree.subjects("train") == ["sub-01"]
        assert tree.subjects("test") == ["sub-01"]
        assert tree.n_files("train") == 1
        assert tree.n_files("test") == 1

    def test_resolves_through_relative_links(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        root = tmp_path / "splits"
        link_subject(root, "train", "sub-01", corpus)
        write_manifest(root, subject_level=True)
        tree = load_split_tree(root)
        files = list(iter_part_files(tree, "train"))
        assert files and all(f.exists() for f in files)

    def test_accepts_a_file_inside_the_tree(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        root = tmp_path / "splits"
        link_subject(root, "train", "sub-01", corpus)
        write_manifest(root, subject_level=True)
        assert load_split_tree(root / "split.json").root == root

    def test_missing_tree_says_how_to_create_one(self, tmp_path: Path) -> None:
        with pytest.raises(SplitTreeError, match="no split tree") as exc:
            load_split_tree(tmp_path / "absent")
        assert "split --dataset" in str(exc.value)

    def test_empty_directory_is_an_error(self, tmp_path: Path) -> None:
        """An empty split would train on nothing and look like a model failure."""
        (tmp_path / "empty").mkdir()
        with pytest.raises(SplitTreeError, match="no parts"):
            load_split_tree(tmp_path / "empty")

    def test_invalid_manifest_reports_clearly(self, tmp_path: Path) -> None:
        root = tmp_path / "splits"
        (root / "train").mkdir(parents=True)
        (root / "split.json").write_text("{not json")
        with pytest.raises(SplitTreeError, match="not valid JSON"):
            load_split_tree(root)

    def test_dataset_layer_is_parsed_in_merged_trees(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        root = tmp_path / "splits"
        link_subject(root, "train", "sub-01", corpus, dataset="chisco")
        link_subject(root, "train", "sub-01", corpus / "other", dataset="chineseeeg")
        write_manifest(root, subject_level=True)
        tree = load_split_tree(root)
        datasets = {entry.dataset for entry in tree.parts["train"]}
        assert datasets == {"chisco", "chineseeeg"}


class TestLeakageDetection:
    def test_same_recording_in_two_parts_is_rejected(self, tmp_path: Path) -> None:
        """The failure mode that makes a model look good while generalising badly."""
        corpus = tmp_path / "corpus"
        root = tmp_path / "splits"
        link_subject(root, "train", "sub-01", corpus)
        link_subject(root, "test", "sub-01", corpus)
        write_manifest(root, subject_level=True)
        tree = load_split_tree(root)
        with pytest.raises(SplitTreeError, match="leaks data across parts"):
            check_no_leakage(tree)

    def test_disjoint_parts_pass(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        root = tmp_path / "splits"
        link_subject(root, "train", "sub-01", corpus)
        link_subject(root, "test", "sub-02", corpus)
        write_manifest(root, subject_level=True)
        check_no_leakage(load_split_tree(root))

    def test_run_level_sharing_a_subject_but_not_runs_is_fine(self, tmp_path: Path) -> None:
        """Run level deliberately shares participants; only files must differ."""
        corpus = tmp_path / "corpus"
        root = tmp_path / "splits"
        link_run(root, "train", "sub-01", "ses-01", "run-01", corpus)
        link_run(root, "test", "sub-01", "ses-01", "run-02", corpus)
        write_manifest(root, subject_level=False)
        check_no_leakage(load_split_tree(root))

    def test_identical_run_number_in_two_sessions_is_not_a_leak(self, tmp_path: Path) -> None:
        """run-01 exists in two sessions; they are different recordings."""
        corpus = tmp_path / "corpus"
        root = tmp_path / "splits"
        link_run(root, "train", "sub-01", "ses-LittlePrince", "run-01", corpus)
        link_run(root, "test", "sub-01", "ses-GarnettDream", "run-01", corpus)
        write_manifest(root, subject_level=False)
        check_no_leakage(load_split_tree(root))


class TestSummarise:
    def test_reports_level_and_counts(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        root = tmp_path / "splits"
        link_subject(root, "train", "sub-01", corpus)
        link_subject(root, "test", "sub-02", corpus)
        write_manifest(root, subject_level=True)
        text = summarise(load_split_tree(root))
        assert "subject-level" in text
        assert "train: 1 subjects" in text

    def test_reports_run_level(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        root = tmp_path / "splits"
        link_run(root, "train", "sub-01", "ses-01", "run-01", corpus)
        write_manifest(root, subject_level=False)
        assert "run-level" in summarise(load_split_tree(root))
