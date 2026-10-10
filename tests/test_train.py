"""Tests for the CTC training and evaluation loop."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch
import torch.nn as nn

from bci_eeg_to_text.data import Trial, collate
from bci_eeg_to_text.model import EEGToTextCTC, ModelConfig
from bci_eeg_to_text.tokenizer import CharTokenizer
from bci_eeg_to_text.train import (
    TrainConfig,
    compute_loss,
    evaluate,
    make_batches,
    move_batch_to_device,
    resolve_device,
    train_one_epoch,
)


@pytest.fixture
def tokenizer() -> CharTokenizer:
    return CharTokenizer.build(["你好", "我想"])


@pytest.fixture
def model() -> EEGToTextCTC:
    return EEGToTextCTC(ModelConfig(n_channels=4, n_classes=4, d_model=8, n_blocks=2, dropout=0.0))


def make_trial(tokenizer: CharTokenizer, text: str, n_samples: int = 512) -> Trial:
    # Seed from a stable digest, not hash(): Python randomises str hashing per
    # process, which made this helper (and any test asserting on random-model
    # numerics) flaky across runs.
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")
    rng = np.random.default_rng(seed)
    return Trial(
        eeg=rng.standard_normal((4, n_samples)).astype(np.float32),
        target=np.array(tokenizer.encode(text), dtype=np.int64),
        text=text,
    )


class TestResolveDevice:
    def test_defaults_to_a_real_device(self) -> None:
        assert resolve_device().type in {"cpu", "cuda"}

    def test_explicit_cpu_is_honoured(self) -> None:
        assert resolve_device("cpu").type == "cpu"

    def test_cuda_request_fails_loudly_when_unavailable(self) -> None:
        """Silently falling back to CPU would hide a broken GPU setup."""
        if torch.cuda.is_available():
            pytest.skip("CUDA is available here")
        with pytest.raises(RuntimeError, match="CUDA was requested but is not available"):
            resolve_device("cuda")


class TestComputeLoss:
    def test_loss_is_finite_and_positive(
        self, model: EEGToTextCTC, tokenizer: CharTokenizer
    ) -> None:
        batch = collate([make_trial(tokenizer, "你好"), make_trial(tokenizer, "我想")])
        moved = move_batch_to_device(batch, torch.device("cpu"))
        loss = compute_loss(model, moved, nn.CTCLoss(blank=tokenizer.blank))
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_loss_decreases_when_target_is_learned(self, tokenizer: CharTokenizer) -> None:
        """Sanity check that the CTC plumbing can actually be optimized."""
        torch.manual_seed(0)
        model = EEGToTextCTC(
            ModelConfig(n_channels=4, n_classes=len(tokenizer), d_model=16, n_blocks=2, dropout=0.0)
        )
        # Give every trial the same input and the same target so the model can
        # memorize it; the loss must then fall.
        fixed = np.ones((4, 1024), dtype=np.float32)
        trials = [
            Trial(eeg=fixed, target=np.array(tokenizer.encode("你好")), text="你好")
            for _ in range(4)
        ]
        batches = make_batches(trials, 4)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        criterion = nn.CTCLoss(blank=tokenizer.blank)
        first = float(
            compute_loss(model, move_batch_to_device(batches[0], torch.device("cpu")), criterion)
        )
        for _ in range(25):
            train_one_epoch(
                model, batches, optimizer, tokenizer, torch.device("cpu"), TrainConfig()
            )
        last = float(
            compute_loss(model, move_batch_to_device(batches[0], torch.device("cpu")), criterion)
        )
        assert last < first

    def test_loss_with_padding_still_bounded(
        self, model: EEGToTextCTC, tokenizer: CharTokenizer
    ) -> None:
        """input_lengths must keep CTC from reading padded frames."""
        batch = collate([make_trial(tokenizer, "你好", 512), make_trial(tokenizer, "我想", 512)])
        moved = move_batch_to_device(batch, torch.device("cpu"))
        loss = compute_loss(model, moved, nn.CTCLoss(blank=tokenizer.blank))
        assert torch.isfinite(loss)


class TestTrainOneEpoch:
    def test_returns_finite_mean_loss(self, model: EEGToTextCTC, tokenizer: CharTokenizer) -> None:
        batches = make_batches([make_trial(tokenizer, "你好"), make_trial(tokenizer, "我想")], 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss = train_one_epoch(
            model, batches, optimizer, tokenizer, torch.device("cpu"), TrainConfig()
        )
        assert np.isfinite(loss)

    def test_parameters_change_after_a_step(self, tokenizer: CharTokenizer) -> None:
        torch.manual_seed(0)
        model = EEGToTextCTC(
            ModelConfig(n_channels=4, n_classes=len(tokenizer), d_model=8, n_blocks=2, dropout=0.0)
        )
        before = [p.detach().clone() for p in model.parameters()]
        batches = make_batches([make_trial(tokenizer, "你好")], 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        train_one_epoch(model, batches, optimizer, tokenizer, torch.device("cpu"), TrainConfig())
        assert any(
            not torch.equal(b, p.detach()) for b, p in zip(before, model.parameters(), strict=True)
        )

    def test_empty_batches_does_not_crash(
        self, model: EEGToTextCTC, tokenizer: CharTokenizer
    ) -> None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss = train_one_epoch(model, [], optimizer, tokenizer, torch.device("cpu"), TrainConfig())
        assert loss == 0.0

    def test_gradient_clipping_is_applied(self, tokenizer: CharTokenizer) -> None:
        """A huge learning rate must not produce non-finite weights."""
        torch.manual_seed(0)
        model = EEGToTextCTC(
            ModelConfig(n_channels=4, n_classes=len(tokenizer), d_model=8, n_blocks=2, dropout=0.0)
        )
        batches = make_batches([make_trial(tokenizer, "你好")], 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e6)
        train_one_epoch(
            model,
            batches,
            optimizer,
            tokenizer,
            torch.device("cpu"),
            TrainConfig(grad_clip=1.0),
        )
        assert all(torch.isfinite(p).all() for p in model.parameters())


class TestEvaluate:
    def test_reports_metrics_and_examples(
        self, model: EEGToTextCTC, tokenizer: CharTokenizer
    ) -> None:
        batches = make_batches([make_trial(tokenizer, "你好"), make_trial(tokenizer, "我想")], 2)
        metrics = evaluate(model, batches, tokenizer, torch.device("cpu"))
        assert metrics.n_examples == 2
        assert 0.0 <= metrics.sentence_accuracy <= 1.0
        assert np.isfinite(metrics.cer)
        assert len(metrics.examples) == 2

    def test_does_not_update_weights(self, model: EEGToTextCTC, tokenizer: CharTokenizer) -> None:
        batches = make_batches([make_trial(tokenizer, "你好")], 1)
        before = [p.detach().clone() for p in model.parameters()]
        evaluate(model, batches, tokenizer, torch.device("cpu"))
        assert all(
            torch.equal(b, p.detach()) for b, p in zip(before, model.parameters(), strict=True)
        )

    def test_evaluation_is_deterministic(
        self, model: EEGToTextCTC, tokenizer: CharTokenizer
    ) -> None:
        batches = make_batches([make_trial(tokenizer, "你好")], 1)
        a = evaluate(model, batches, tokenizer, torch.device("cpu"))
        b = evaluate(model, batches, tokenizer, torch.device("cpu"))
        assert a.examples == b.examples

    def test_empty_input_yields_zero_examples(
        self, model: EEGToTextCTC, tokenizer: CharTokenizer
    ) -> None:
        metrics = evaluate(model, [], tokenizer, torch.device("cpu"))
        assert metrics.n_examples == 0
        assert metrics.sentence_accuracy == 0.0

    def test_caps_reported_examples(self, model: EEGToTextCTC, tokenizer: CharTokenizer) -> None:
        batches = make_batches([make_trial(tokenizer, "你好")] * 10, 10)
        metrics = evaluate(model, batches, tokenizer, torch.device("cpu"), max_examples=3)
        assert len(metrics.examples) == 3


class TestMakeBatches:
    def test_splits_into_chunks(self, tokenizer: CharTokenizer) -> None:
        trials = [make_trial(tokenizer, "你好")] * 5
        assert [len(b["texts"]) for b in make_batches(trials, 2)] == [2, 2, 1]

    def test_no_empty_batch_for_empty_input(self, tokenizer: CharTokenizer) -> None:
        assert make_batches([], 4) == []

    def test_shuffle_is_deterministic_for_a_seed(self, tokenizer: CharTokenizer) -> None:
        trials = [make_trial(tokenizer, t) for t in ("你好", "我想")] * 4
        a = [list(b["texts"]) for b in make_batches(trials, 2, shuffle=True, seed=1)]
        b = [list(b["texts"]) for b in make_batches(trials, 2, shuffle=True, seed=1)]
        assert a == b

    def test_every_trial_appears_exactly_once(self, tokenizer: CharTokenizer) -> None:
        trials = [make_trial(tokenizer, t) for t in ("你好", "我想")] * 3
        texts = [str(t) for b in make_batches(trials, 2, shuffle=True, seed=3) for t in b["texts"]]
        assert sorted(texts) == sorted(t.text for t in trials)
