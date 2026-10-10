"""Tests for the command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bci_eeg_to_text.backbones import TRAINING_MODES
from bci_eeg_to_text.cli import build_parser, cmd_inspect, main


def write_cache(path: Path, n_trials: int, text: str = "你好") -> Path:
    """Write a cache in the format extract_run produces."""
    rng = np.random.default_rng(0)
    # Long enough that the x512 conv stride still leaves several CTC frames.
    trials = [rng.standard_normal((4, 2048)).astype(np.float32) for _ in range(n_trials)]
    flat = np.concatenate(trials, axis=1)
    offsets = np.zeros(n_trials + 1, dtype=np.int64)
    np.cumsum([t.shape[1] for t in trials], out=offsets[1:])
    np.savez_compressed(
        path,
        flat=flat,
        offsets=offsets,
        n_channels=np.array(4),
        texts=np.array([text] * n_trials, dtype=object),
        lengths=np.array([t.shape[1] for t in trials], dtype=np.int64),
        onsets=np.arange(n_trials),
        sample_rate=np.array(125.0),
        subject=np.array("sub-01"),
        run=np.array(1),
        task=np.array("imagine"),
    )
    return path


class TestParser:
    def test_requires_a_subcommand(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    @pytest.mark.parametrize("command", ["prepare", "inspect", "train", "evaluate"])
    def test_subcommands_exist(self, command: str) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([command, "--help"])

    def test_data_root_defaults_to_the_environment(self) -> None:
        """Nothing about the dataset layout is hard-coded in the parser.

        The flag defaults to None so DataConfig can fall back to
        EEG_TEXT_BCI_DATA_ROOT and then a built-in default.
        """
        args = build_parser().parse_args(["prepare", "--out", "x"])
        assert args.data_root is None
        assert args.data_subdir is None
        assert args.data_writable is None

    def test_data_location_flags_are_available(self) -> None:
        args = build_parser().parse_args(
            [
                "prepare",
                "--out",
                "x",
                "--data-root",
                "/data",
                "--data-subdir",
                "corpora",
                "--data-writable",
            ]
        )
        assert args.data_root == "/data"
        assert args.data_subdir == "corpora"
        assert args.data_writable is True

    def test_dataset_defaults_to_chisco(self) -> None:
        args = build_parser().parse_args(["prepare", "--out", "x"])
        assert args.dataset == "chisco"

    def test_dataset_choices_include_merged(self) -> None:
        for choice in ("chisco", "chineseeeg", "merged"):
            args = build_parser().parse_args(["prepare", "--out", "x", "--dataset", choice])
            assert args.dataset == choice

    def test_split_defaults_to_subject_level(self) -> None:
        """Run-level splitting must be opt-in, never the default."""
        args = build_parser().parse_args(["split"])
        assert args.level == "subject"

    def test_decimate_defaults_to_downsampling(self) -> None:
        args = build_parser().parse_args(["prepare", "--out", "x"])
        assert args.decimate == 8


class TestInspect:
    def test_missing_file_fails_cleanly(self, tmp_path: Path) -> None:
        args = build_parser().parse_args(["inspect", str(tmp_path / "nope.edf")])
        with pytest.raises(FileNotFoundError):
            cmd_inspect(args)

    def test_reports_missing_data_for_annex_pointer(self, tmp_path: Path) -> None:
        """A pointer stub must not surface as a confusing EDF parse error."""
        from bci_eeg_to_text.eeg import MissingDataError

        edf = tmp_path / "sub-01_ses-01_task-imagine_run-01_eeg.edf"
        edf.write_text("/annex/objects/SHA256E-s123--abc.edf\n")
        args = build_parser().parse_args(["inspect", str(edf)])
        with pytest.raises(MissingDataError, match="unfetched git-annex pointer"):
            cmd_inspect(args)


class TestPrepareErrors:
    def test_no_runs_reports_cleanly(self, tmp_path: Path, capsys) -> None:
        """An empty data root must not crash; it reports and returns 0.

        Missing data is an expected condition (the corpus is partially
        downloaded by design), so this is not treated as a hard failure.
        """
        code = main(["prepare", "--data-root", str(tmp_path), "--out", str(tmp_path / "o")])
        assert code == 0
        assert "no usable runs" in capsys.readouterr().err


class TestTrainAndEvaluateRequireCache:
    def test_train_without_cache_exits_with_message(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit, match="run `prepare` first"):
            main(
                [
                    "train",
                    "--cache",
                    str(tmp_path / "empty"),
                    "--out",
                    str(tmp_path / "runs"),
                ]
            )
        (tmp_path / "empty").mkdir()
        with pytest.raises(SystemExit, match="run `prepare` first"):
            main(
                [
                    "train",
                    "--cache",
                    str(tmp_path / "empty"),
                    "--out",
                    str(tmp_path / "runs"),
                ]
            )


class TestEndToEndOnSyntheticCache:
    """Train and evaluate through the CLI without touching the real dataset."""

    @pytest.fixture
    def cache_dir(self, tmp_path: Path) -> Path:
        cache = tmp_path / "trials"
        cache.mkdir()
        # Repeated text keeps the vocabulary tiny so the smoke run is fast.
        write_cache(cache / "run-01.npz", 4, "你好")
        return cache

    def test_train_writes_checkpoint_and_vocab(
        self, cache_dir: Path, tmp_path: Path, capsys
    ) -> None:
        out = tmp_path / "runs"
        code = main(
            [
                "train",
                "--cache",
                str(cache_dir),
                "--out",
                str(out),
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--n-samples",
                "2048",
                "--d-model",
                "16",
                "--n-blocks",
                "2",
                "--device",
                "cpu",
                "--max-steps",
                "1",
            ]
        )
        assert code == 0
        assert (out / "last.pt").exists()
        assert (out / "vocab.json").exists()
        assert "device=cpu" in capsys.readouterr().out

    def test_evaluate_reports_metrics(self, cache_dir: Path, tmp_path: Path, capsys) -> None:
        out = tmp_path / "runs"
        main(
            [
                "train",
                "--cache",
                str(cache_dir),
                "--out",
                str(out),
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--n-samples",
                "2048",
                "--d-model",
                "16",
                "--n-blocks",
                "2",
                "--device",
                "cpu",
            ]
        )
        capsys.readouterr()
        code = main(
            [
                "evaluate",
                "--cache",
                str(cache_dir),
                "--checkpoint",
                str(out / "last.pt"),
                "--device",
                "cpu",
                "--n-samples",
                "2048",
            ]
        )
        assert code == 0

    def test_evaluate_with_beam_search_runs(self, cache_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "runs"
        main(
            [
                "train",
                "--cache",
                str(cache_dir),
                "--out",
                str(out),
                "--epochs",
                "1",
                "--n-samples",
                "2048",
                "--d-model",
                "16",
                "--n-blocks",
                "2",
                "--device",
                "cpu",
            ]
        )
        code = main(
            [
                "evaluate",
                "--cache",
                str(cache_dir),
                "--checkpoint",
                str(out / "last.pt"),
                "--device",
                "cpu",
                "--n-samples",
                "2048",
                "--beam-width",
                "3",
            ]
        )
        assert code == 0

    def test_vocab_file_loads_back(self, cache_dir: Path, tmp_path: Path) -> None:
        from bci_eeg_to_text.tokenizer import CharTokenizer

        out = tmp_path / "runs"
        main(
            [
                "train",
                "--cache",
                str(cache_dir),
                "--out",
                str(out),
                "--epochs",
                "1",
                "--n-samples",
                "2048",
                "--d-model",
                "16",
                "--n-blocks",
                "2",
                "--device",
                "cpu",
            ]
        )
        tokenizer = CharTokenizer.load(out / "vocab.json")
        assert tokenizer.decode(tokenizer.encode("你好")) == "你好"

    def test_checkpoint_vocab_matches_model_output_classes(
        self, cache_dir: Path, tmp_path: Path
    ) -> None:
        """A mismatch here would make decoding produce garbage silently."""
        import torch

        out = tmp_path / "runs"
        main(
            [
                "train",
                "--cache",
                str(cache_dir),
                "--out",
                str(out),
                "--epochs",
                "1",
                "--n-samples",
                "2048",
                "--d-model",
                "16",
                "--n-blocks",
                "2",
                "--device",
                "cpu",
            ]
        )
        checkpoint = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        # The checkpoint must carry everything evaluate() needs to rebuild the
        # exact architecture; otherwise a reload silently mismatches the head.
        assert checkpoint["n_classes"] == len(checkpoint["vocab"])
        assert checkpoint["mode"] in TRAINING_MODES
        assert "backbone_kwargs" in checkpoint
        assert checkpoint["n_channels"] > 0

    def test_training_report_is_valid_json_metrics(
        self, cache_dir: Path, tmp_path: Path, capsys
    ) -> None:
        out = tmp_path / "runs"
        main(
            [
                "train",
                "--cache",
                str(cache_dir),
                "--out",
                str(out),
                "--epochs",
                "1",
                "--n-samples",
                "2048",
                "--d-model",
                "16",
                "--n-blocks",
                "2",
                "--device",
                "cpu",
            ]
        )
        main(
            [
                "evaluate",
                "--cache",
                str(cache_dir),
                "--checkpoint",
                str(out / "last.pt"),
                "--device",
                "cpu",
                "--n-samples",
                "2048",
            ]
        )
        out_text = capsys.readouterr().out
        start = out_text.index("{")
        end = out_text.index("}", start) + 1
        payload = json.loads(out_text[start:end])
        assert set(payload) >= {"loss", "cer", "sentence_accuracy", "n_examples"}


class TestSplitCommand:
    """The split command must expose the agreed, gender-balanced policy."""

    @pytest.fixture(autouse=True)
    def scratch_cwd(self, tmp_path, monkeypatch):
        """Run in a scratch directory: --link writes a symlink tree.

        The dataset is re-pointed with an absolute path because the default is
        relative to the repository, which is no longer the working directory.
        """
        repo = Path(__file__).resolve().parents[1]
        data = repo / "eeg-text-bci-data" / "bci-eeg"
        monkeypatch.chdir(tmp_path)
        if data.exists():
            monkeypatch.setenv("EEG_TEXT_BCI_DATA_ROOT", str(repo / "eeg-text-bci-data"))
        return tmp_path

    def test_chisco_uses_one_to_one(self, capsys) -> None:
        code = main(["split", "--dataset", "chisco"])
        if code != 0:
            pytest.skip("dataset not mounted")
        out = capsys.readouterr().out
        assert "'train': 1, 'test': 1" in out
        assert "1F/1M" in out

    def test_chineseeeg_uses_three_one_one(self, capsys) -> None:
        code = main(["split", "--dataset", "chineseeeg"])
        if code != 0:
            pytest.skip("dataset not mounted")
        out = capsys.readouterr().out
        assert "'train': 3, 'valid': 1, 'test': 1" in out
        assert "3F/3M" in out

    def test_ratios_can_be_overridden(self, capsys) -> None:
        code = main(["split", "--dataset", "chineseeeg", "--ratios", "train=1,test=1"])
        if code != 0:
            pytest.skip("dataset not mounted")
        assert "'train': 1, 'test': 1" in capsys.readouterr().out

    def test_run_level_warns_on_stderr(self, tmp_path, monkeypatch, capsys) -> None:
        # Materialising links writes files, so run in a scratch directory.
        monkeypatch.chdir(tmp_path)
        code = main(["split", "--dataset", "chisco", "--level", "run"])
        if code != 0:
            pytest.skip("dataset not mounted")
        err = capsys.readouterr().err
        assert "NOT a cross-subject estimate" in err

    def test_subject_level_does_not_warn(self, capsys) -> None:
        code = main(["split", "--dataset", "chisco"])
        if code != 0:
            pytest.skip("dataset not mounted")
        assert "NOT a cross-subject" not in capsys.readouterr().err

    def test_writes_json_when_asked(self, tmp_path, capsys) -> None:
        out = tmp_path / "split.json"
        code = main(["split", "--dataset", "chisco", "--out", str(out)])
        if code != 0:
            pytest.skip("dataset not mounted")
        pytest.importorskip("json")
        payload = json.loads(out.read_text())
        assert payload["level"] == "subject"
        assert payload["subject_level"] is True
        assert set(payload["parts"]) == {"train", "test"}

    def test_rejects_malformed_ratios(self, tmp_path: Path) -> None:
        """Must fail identically whether or not the dataset is mounted.

        Validation therefore happens before the data root is touched; otherwise
        an unmounted dataset would mask the bad argument.
        """
        with pytest.raises(SystemExit, match="expected name=int"):
            main(
                [
                    "split",
                    "--dataset",
                    "chisco",
                    "--ratios",
                    "bogus",
                    "--data-root",
                    str(tmp_path / "absent"),
                ]
            )
