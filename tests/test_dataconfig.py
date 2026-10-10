"""Tests for dataset-location config and symlinked split materialisation.

The design goals under test:

* the corpus subdirectory is never hard-coded, and
* a split is a browsable tree of **relative** symlinks that costs no disk space
  and keeps working when the dataset and the split move together.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from bci_eeg_to_text.dataconfig import (
    DATA_ROOT_ENV,
    DATA_SUBDIR_ENV,
    DATA_WRITABLE_ENV,
    DEFAULT_DATA_SUBDIR,
    SPLIT_DIRNAME,
    DataConfig,
    load_split_manifest,
    materialise_split,
    relative_symlink,
)
from bci_eeg_to_text.splits import Participant, make_splits


def make_corpus(
    root: Path, names: list[str], subdir: str = "bci-eeg", corpus: str = "ds005170"
) -> Path:
    """Create a minimal BIDS-like corpus with one run per subject."""
    corpus_root = root / subdir / corpus
    for name in names:
        eeg = corpus_root / name / "ses-01" / "eeg"
        eeg.mkdir(parents=True)
        (eeg / f"{name}_ses-01_task-imagine_run-01_eeg.edf").write_bytes(b"0" * 32)
        (eeg / f"{name}_ses-01_task-imagine_run-01_eeg.json").write_text("{}")
        (eeg / f"{name}_ses-01_task-imagine_run-01_events.tsv").write_text("onset\n")
    return corpus_root


@pytest.fixture(autouse=True)
def clean_env(monkeypatch) -> None:
    """Isolate tests from the ambient environment."""
    for name in (DATA_ROOT_ENV, DATA_SUBDIR_ENV, DATA_WRITABLE_ENV):
        monkeypatch.delenv(name, raising=False)


class TestDataConfigFromEnv:
    def test_defaults_when_environment_is_empty(self) -> None:
        config = DataConfig.from_env()
        assert config.subdir == DEFAULT_DATA_SUBDIR
        assert config.assume_writable is False

    def test_reads_root_from_environment(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
        assert DataConfig.from_env().root == tmp_path

    def test_reads_subdir_from_environment(self, monkeypatch) -> None:
        """The corpus subdirectory must be configurable, not hard-coded."""
        monkeypatch.setenv(DATA_SUBDIR_ENV, "corpora")
        assert DataConfig.from_env().subdir == "corpora"

    def test_reads_writable_flag(self, monkeypatch) -> None:
        monkeypatch.setenv(DATA_WRITABLE_ENV, "true")
        assert DataConfig.from_env().assume_writable is True

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
    def test_accepts_common_true_spellings(self, monkeypatch, value: str) -> None:
        monkeypatch.setenv(DATA_WRITABLE_ENV, value)
        assert DataConfig.from_env().assume_writable is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
    def test_accepts_common_false_spellings(self, monkeypatch, value: str) -> None:
        monkeypatch.setenv(DATA_WRITABLE_ENV, value)
        assert DataConfig.from_env().assume_writable is False

    def test_explicit_arguments_beat_the_environment(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setenv(DATA_ROOT_ENV, "/from/env")
        monkeypatch.setenv(DATA_SUBDIR_ENV, "from-env")
        monkeypatch.setenv(DATA_WRITABLE_ENV, "true")
        config = DataConfig.from_env(root=tmp_path, subdir="explicit", assume_writable=False)
        assert (config.root, config.subdir, config.assume_writable) == (
            tmp_path,
            "explicit",
            False,
        )

    def test_writable_defaults_to_false(self) -> None:
        """Writing into a read-only mount must never happen by accident."""
        assert DataConfig.from_env().assume_writable is False


class TestPaths:
    def test_corpora_dir_uses_the_configured_subdir(self, tmp_path: Path) -> None:
        config = DataConfig(root=tmp_path, subdir="my-corpora")
        assert config.corpora_dir == tmp_path / "my-corpora"

    def test_dataset_path_is_not_hardcoded(self, tmp_path: Path) -> None:
        config = DataConfig(root=tmp_path, subdir="whatever")
        assert config.dataset_path("ds005170") == tmp_path / "whatever" / "ds005170"

    def test_split_root_inside_root_when_writable(self, tmp_path: Path) -> None:
        config = DataConfig(root=tmp_path, subdir="c", assume_writable=True)
        assert config.split_root() == tmp_path / "c" / SPLIT_DIRNAME

    def test_split_root_in_cwd_when_not_writable(self, tmp_path: Path) -> None:
        """Default: never write under the dataset root."""
        config = DataConfig(root=tmp_path / "data", subdir="c", assume_writable=False)
        assert config.split_root(cwd=tmp_path) == tmp_path / SPLIT_DIRNAME

    def test_is_inside_root(self, tmp_path: Path) -> None:
        config = DataConfig(root=tmp_path / "data")
        assert config.is_inside_root(tmp_path / "data" / "c" / "x")
        assert not config.is_inside_root(tmp_path / "elsewhere")


class TestRelativeSymlink:
    def test_creates_a_relative_target(self, tmp_path: Path) -> None:
        target = tmp_path / "data" / "sub-01"
        target.mkdir(parents=True)
        link = tmp_path / "splits" / "train" / "sub-01"
        relative_symlink(target, link)
        assert link.is_symlink()
        assert not os.path.isabs(os.readlink(link))
        assert link.resolve() == target.resolve()

    def test_replaces_an_existing_link(self, tmp_path: Path) -> None:
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        link = tmp_path / "link"
        relative_symlink(a, link)
        relative_symlink(b, link)
        assert link.resolve() == b.resolve()

    def test_no_bytes_are_copied(self, tmp_path: Path) -> None:
        """A 1 MB payload must not be duplicated by linking."""
        payload = tmp_path / "data" / "big.edf"
        payload.parent.mkdir(parents=True)
        payload.write_bytes(b"x" * (1 << 20))
        link = tmp_path / "splits" / "train" / "big.edf"
        relative_symlink(payload, link)
        assert link.stat().st_size == 1 << 20  # same file through the link
        assert not link.is_symlink() or link.resolve() == payload


class TestMaterialiseSplit:
    def test_creates_one_link_per_subject_at_subject_level(self, tmp_path: Path) -> None:
        corpus = make_corpus(tmp_path, ["sub-01", "sub-02"])
        config = DataConfig(root=tmp_path, assume_writable=True)
        split = make_splits(
            [Participant("sub-01", sex="F"), Participant("sub-02", sex="M")],
            ratios={"train": 1, "test": 1},
        )
        result = materialise_split(split, data=config, corpus_paths={"bci-eeg": corpus})
        assert result.n_links == 2
        assert (result.path / "train").is_dir()

    def test_links_are_relative_and_resolve(self, tmp_path: Path) -> None:
        corpus = make_corpus(tmp_path, ["sub-01"])
        config = DataConfig(root=tmp_path, assume_writable=True)
        split = make_splits([Participant("sub-01", sex="F")], ratios={"train": 1})
        result = materialise_split(split, data=config, corpus_paths={"bci-eeg": corpus})
        links = [p for p in result.path.rglob("*") if p.is_symlink()]
        assert links
        for link in links:
            assert not os.path.isabs(os.readlink(link))
            assert link.resolve().exists()

    def test_tree_survives_being_moved_with_the_dataset(self, tmp_path: Path) -> None:
        """Relative links are what make this portable across remounts."""
        corpus = make_corpus(tmp_path, ["sub-01"])
        config = DataConfig(root=tmp_path, assume_writable=True)
        split = make_splits([Participant("sub-01", sex="F")], ratios={"train": 1})
        result = materialise_split(split, data=config, corpus_paths={"bci-eeg": corpus})

        # Remember where the split actually went before moving the tree,
        # rather than assuming root/splits (it honours the corpus subdir).
        relative_split = result.path.relative_to(tmp_path)
        moved = tmp_path.parent / f"{tmp_path.name}-moved"
        shutil.move(str(tmp_path), str(moved))
        moved_split = moved / relative_split
        # At subject level the split links whole directories, and rglob does not
        # descend through symlinked directories, so enumerate explicitly.
        links = [p for p in moved_split.iterdir() if p.is_symlink()]
        links += [
            p
            for part in moved_split.iterdir()
            if part.is_dir()
            for p in part.iterdir()
            if p.is_symlink()
        ]
        assert links, "expected symlinks in the relocated tree"
        assert all(link.resolve().exists() for link in links), (
            "relative symlinks must still resolve after the tree is relocated"
        )

    def test_writes_a_manifest_alongside_the_tree(self, tmp_path: Path) -> None:
        corpus = make_corpus(tmp_path, ["sub-01"])
        config = DataConfig(root=tmp_path, subdir="bci-eeg", assume_writable=True)
        split = make_splits([Participant("sub-01", sex="F")], ratios={"train": 1})
        result = materialise_split(split, data=config, corpus_paths={"bci-eeg": corpus})
        manifest = load_split_manifest(result.path)
        assert manifest["subject_level"] is True
        assert manifest["data_subdir"] == "bci-eeg"
        assert "train" in manifest["parts"]

    def test_does_not_write_inside_root_when_not_writable(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        corpus = make_corpus(tmp_path / "data", ["sub-01"])
        config = DataConfig(root=tmp_path / "data", assume_writable=False)
        split = make_splits([Participant("sub-01", sex="F")], ratios={"train": 1})
        result = materialise_split(
            split, data=config, corpus_paths={"bci-eeg": corpus}, cwd=workspace
        )
        assert result.path == workspace / SPLIT_DIRNAME
        assert not config.is_inside_root(result.path)

    def test_links_every_file_of_a_run_at_run_level(self, tmp_path: Path) -> None:
        """A run is incomplete without its sidecars."""
        corpus = make_corpus(tmp_path, ["sub-01"])
        config = DataConfig(root=tmp_path, assume_writable=True)
        split = make_splits(
            [Participant("sub-01", sex="F")],
            ratios={"train": 1},
            level="run",
            runs_per_subject={"sub-01": ["run-01"]},
        )
        result = materialise_split(split, data=config, corpus_paths={"bci-eeg": corpus})
        assert result.n_links >= 3, "edf, json and events sidecars should all be linked"
        assert any(p.suffix == ".json" for p in result.path.rglob("*.json"))

    def test_records_skipped_entries_instead_of_failing(self, tmp_path: Path) -> None:
        """A subject with no local files must not abort the whole split."""
        config = DataConfig(root=tmp_path, assume_writable=True)
        (tmp_path / "bci-eeg" / "ds005170").mkdir(parents=True)
        split = make_splits([Participant("sub-99", sex="F")], ratios={"train": 1})
        result = materialise_split(
            split,
            data=config,
            corpus_paths={"bci-eeg": tmp_path / "bci-eeg" / "ds005170"},
        )
        assert result.skipped
        assert result.n_links == 0

    def test_merged_corpora_are_separated_by_dataset(self, tmp_path: Path) -> None:
        a = make_corpus(tmp_path, ["sub-01"], corpus="ds005170")
        b = make_corpus(tmp_path, ["sub-01"], corpus="ds004952")
        config = DataConfig(root=tmp_path, assume_writable=True)
        split = make_splits(
            [
                Participant("chisco:sub-01", "chisco", "F"),
                Participant("chineseeeg:sub-01", "chineseeeg", "F"),
            ],
            ratios={"train": 1},
        )
        result = materialise_split(
            split,
            data=config,
            corpus_paths={"chisco": a, "chineseeeg": b},
        )
        assert (result.path / "train" / "chisco").exists()
        assert (result.path / "train" / "chineseeeg").exists()

    def test_link_count_costs_no_meaningful_disk_space(self, tmp_path: Path) -> None:
        """Links must not duplicate the payload."""
        corpus = make_corpus(tmp_path, ["sub-01"])
        big = corpus / "sub-01" / "ses-01" / "eeg" / "sub-01_ses-01_task-imagine_run-01_eeg.edf"
        big.write_bytes(b"x" * (1 << 20))
        config = DataConfig(root=tmp_path, assume_writable=True)
        split = make_splits([Participant("sub-01", sex="F")], ratios={"train": 1})
        result = materialise_split(split, data=config, corpus_paths={"bci-eeg": corpus})
        linked = result.path / "train" / "sub-01"
        assert linked.resolve() == (corpus / "sub-01").resolve()


class TestReadOnlyRoot:
    """`assume_writable` is a user assertion and must fail clearly when wrong."""

    def test_raises_actionable_error_on_read_only_root(self, tmp_path, monkeypatch) -> None:
        from bci_eeg_to_text.dataconfig import DataAccessError

        corpus = make_corpus(tmp_path, ["sub-01"])
        config = DataConfig(root=tmp_path, subdir="bci-eeg", assume_writable=True)
        split = make_splits([Participant("sub-01", sex="F")], ratios={"train": 1})

        # Simulate a read-only mount by making mkdir fail.
        real_mkdir = Path.mkdir

        def deny(self, *args, **kwargs):
            if "splits" in str(self):
                raise OSError(30, "Read-only file system")
            return real_mkdir(self, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", deny)
        with pytest.raises(DataAccessError) as exc:
            materialise_split(split, data=config, corpus_paths={"bci-eeg": corpus})
        message = str(exc.value)
        assert "read-only" in message
        assert "--data-writable" in message, "the message must name the flag to drop"

    def test_non_writable_config_avoids_the_root_entirely(self, tmp_path) -> None:
        """The safe default never touches the dataset root."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        corpus = make_corpus(tmp_path / "data", ["sub-01"])
        config = DataConfig(root=tmp_path / "data", assume_writable=False)
        split = make_splits([Participant("sub-01", sex="F")], ratios={"train": 1})
        result = materialise_split(
            split, data=config, corpus_paths={"bci-eeg": corpus}, cwd=workspace
        )
        assert workspace in result.path.parents
