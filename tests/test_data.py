"""Tests for the trial-building pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bci_eeg_to_text.data import ExtractConfig, TrialDataset, _fit_length, collate, discover_runs
from bci_eeg_to_text.tokenizer import CharTokenizer


def write_cache(path: Path, trials: list[np.ndarray], texts: list[str]) -> Path:
    """Write a cache file in the format extract_run produces.

    Trials have differing lengths, so they are stored concatenated along time
    with an offsets array rather than stacked.
    """
    flat = np.concatenate([t.astype(np.float32) for t in trials], axis=1)
    offsets = np.zeros(len(trials) + 1, dtype=np.int64)
    np.cumsum([t.shape[1] for t in trials], out=offsets[1:])
    np.savez_compressed(
        path,
        flat=flat,
        offsets=offsets,
        n_channels=np.array(trials[0].shape[0]),
        texts=np.array(texts, dtype=object),
        lengths=np.array([t.shape[1] for t in trials], dtype=np.int64),
        onsets=np.arange(len(trials)),
        sample_rate=np.array(1000.0),
        subject=np.array("sub-01"),
        run=np.array(1),
        task=np.array("imagine"),
    )
    return path


class TestFitLength:
    def test_pads_short_trial_with_zeros(self) -> None:
        trial = np.ones((4, 10), dtype=np.float32)
        out = _fit_length(trial, 20)
        assert out.shape == (4, 20)
        assert out[:, :10].sum() == 40
        assert out[:, 10:].sum() == 0

    def test_crops_long_trial(self) -> None:
        trial = np.arange(4 * 30, dtype=np.float32).reshape(4, 30)
        out = _fit_length(trial, 10)
        assert out.shape == (4, 10)
        assert np.array_equal(out, trial[:, :10])

    def test_exact_length_is_unchanged(self) -> None:
        trial = np.ones((2, 5), dtype=np.float32)
        assert np.array_equal(_fit_length(trial, 5), trial)

    def test_preserves_channel_count(self) -> None:
        out = _fit_length(np.ones((128, 7), dtype=np.float32), 32)
        assert out.shape[0] == 128


class TestTrialDataset:
    @pytest.fixture
    def tokenizer(self) -> CharTokenizer:
        return CharTokenizer.build(["你好", "我想"])

    @pytest.fixture
    def cache(self, tmp_path: Path) -> Path:
        rng = np.random.default_rng(0)
        trials = [rng.standard_normal((4, 100 + i * 10)).astype(np.float32) for i in range(3)]
        return write_cache(tmp_path / "run1.npz", trials, ["你好", "我想", "你好"])

    def test_length_counts_all_trials(self, cache: Path, tokenizer: CharTokenizer) -> None:
        assert len(TrialDataset([cache], tokenizer, n_samples=64)) == 3

    def test_item_has_fixed_length_eeg(self, cache: Path, tokenizer: CharTokenizer) -> None:
        ds = TrialDataset([cache], tokenizer, n_samples=64)
        assert ds[0].eeg.shape == (4, 64)

    def test_target_encodes_the_text(self, cache: Path, tokenizer: CharTokenizer) -> None:
        ds = TrialDataset([cache], tokenizer, n_samples=64)
        item = ds[0]
        assert tokenizer.decode(item.target.tolist()) == item.text == "你好"

    def test_target_length_is_bounded(self, cache: Path, tokenizer: CharTokenizer) -> None:
        ds = TrialDataset([cache], tokenizer, n_samples=64, max_target_length=1)
        assert all(len(ds[i].target) <= 1 for i in range(len(ds)))

    def test_target_is_never_empty(self, cache: Path, tokenizer: CharTokenizer) -> None:
        """An empty target makes CTC loss undefined."""
        ds = TrialDataset([cache], tokenizer, n_samples=64)
        assert all(len(ds[i].target) > 0 for i in range(len(ds)))

    def test_multiple_caches_are_concatenated(
        self, cache: Path, tmp_path: Path, tokenizer: CharTokenizer
    ) -> None:
        other = write_cache(tmp_path / "run2.npz", [np.zeros((4, 50), dtype=np.float32)], ["我想"])
        assert len(TrialDataset([cache, other], tokenizer, n_samples=64)) == 4

    def test_items_are_independent_of_iteration(
        self, cache: Path, tokenizer: CharTokenizer
    ) -> None:
        ds = TrialDataset([cache], tokenizer, n_samples=64)
        assert ds[1].text == "我想"
        assert ds[0].text == "你好"


class TestCollate:
    @pytest.fixture
    def tokenizer(self) -> CharTokenizer:
        return CharTokenizer.build(["你好", "我想"])

    def test_shapes_and_lengths(self, tmp_path: Path, tokenizer: CharTokenizer) -> None:
        rng = np.random.default_rng(0)
        cache = write_cache(
            tmp_path / "r.npz",
            [rng.standard_normal((4, 100)).astype(np.float32)] * 3,
            ["你好", "我想", "你好"],
        )
        ds = TrialDataset([cache], tokenizer, n_samples=64)
        batch = collate([ds[0], ds[1]])
        assert batch["eeg"].shape == (2, 4, 64)
        assert batch["target_lengths"].tolist() == [2, 2]
        assert batch["input_lengths"].tolist() == [64, 64]
        assert batch["targets"].tolist() == [*ds[0].target, *ds[1].target]

    def test_target_lengths_match_concatenated_targets(
        self, tmp_path: Path, tokenizer: CharTokenizer
    ) -> None:
        cache = write_cache(
            tmp_path / "r.npz",
            [np.zeros((4, 100), dtype=np.float32)] * 2,
            ["你好", "我想"],
        )
        ds = TrialDataset([cache], tokenizer, n_samples=64)
        batch = collate([ds[0], ds[1]])
        assert sum(batch["target_lengths"]) == len(batch["targets"])


@pytest.mark.realdata
class TestRealDataset:
    def test_extract_run_matches_pulse_count(self, tmp_path: Path) -> None:
        """End-to-end: extract one real run and confirm the trial count."""
        root = Path("eeg-text-bci-data/bci-eeg/ds005170")
        runs = discover_runs(root)
        if not runs:
            pytest.skip("dataset not mounted")
        target = next(
            (r for r in runs if r.name == "sub-01_ses-02_task-imagine_run-010_eeg.edf"), None
        )
        if target is None:
            pytest.skip("run-010 not present")

        from bci_eeg_to_text.data import extract_run

        info = extract_run(
            target,
            root,
            tmp_path / "run010.npz",
            config=ExtractConfig(decimate=8),
        )
        assert info["n_trials"] == 150
        assert info["shape"][0] == 150
        assert info["shape"][1] == 128

    def test_every_run_pulse_count_matches_its_split_file(self) -> None:
        """The alignment contract from issue #1, checked across the dataset.

        This is the single most important invariant in the project: if it fails,
        every downstream number is meaningless.
        """
        import zipfile

        from bci_eeg_to_text.eeg import read_metadata, read_signals

        root = Path("eeg-text-bci-data/bci-eeg/ds005170")
        runs = discover_runs(root)
        if not runs:
            pytest.skip("dataset not mounted")
        # A representative sample keeps this test affordable; the full sweep is
        # done by the prepare CLI and recorded as an artifact.
        sample = runs[::37]
        mismatches = []
        for run in sample:
            meta = read_metadata(run)
            split = root / "textdataset" / f"split_data_{meta.run_id.run}.xlsx"
            if not split.exists():
                mismatches.append((run.name, "missing split file"))
                continue
            n_text = len(
                [
                    ln
                    for ln in zipfile.ZipFile(split)
                    .read("xl/worksheets/sheet1.xml")
                    .decode()
                    .split("<row")
                    if ln
                ][1:]
            )
            signals, _ = read_signals(run, ["Trigger"])
            from bci_eeg_to_text.eeg import find_pulses

            n_pulses = len(find_pulses(signals[0]))
            if n_pulses != n_text:
                mismatches.append((run.name, f"{n_pulses} pulses vs {n_text} sentences"))
        assert not mismatches, f"alignment mismatches: {mismatches}"
