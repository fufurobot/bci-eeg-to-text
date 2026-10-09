"""Tests for the CTC EEG encoder."""

from __future__ import annotations

import pytest
import torch

from bci_eeg_to_text.model import EEGToTextCTC, ModelConfig


@pytest.fixture
def config() -> ModelConfig:
    return ModelConfig(n_channels=8, n_classes=20, d_model=16, n_blocks=3, dropout=0.0)


class TestShapes:
    def test_forward_output_shape(self, config: ModelConfig) -> None:
        model = EEGToTextCTC(config)
        x = torch.randn(2, config.n_channels, 4000)
        out = model(x)
        assert out.ndim == 3
        assert out.shape[0] == 2
        assert out.shape[2] == config.n_classes

    def test_output_is_logits_over_time(self, config: ModelConfig) -> None:
        """CTC expects (batch, time, classes), not (batch, classes, time)."""
        model = EEGToTextCTC(config)
        out = model(torch.randn(1, config.n_channels, 4000))
        assert out.shape[1] > 1
        assert out.shape[2] == config.n_classes

    def test_rejects_wrong_rank(self, config: ModelConfig) -> None:
        model = EEGToTextCTC(config)
        with pytest.raises(ValueError, match=r"expected \(batch, channels, time\)"):
            model(torch.randn(2, 4000))

    def test_rejects_wrong_channel_count(self, config: ModelConfig) -> None:
        model = EEGToTextCTC(config)
        with pytest.raises(ValueError, match="expected 8 channels"):
            model(torch.randn(1, 9, 4000))


class TestDownsampling:
    def test_output_stride_matches_observed_reduction(self, config: ModelConfig) -> None:
        """The reported stride must equal the real reduction, or CTC lengths lie."""
        model = EEGToTextCTC(config).eval()
        x = torch.randn(1, config.n_channels, 8192)
        with torch.no_grad():
            out = model(x)
        ratio = x.shape[-1] // out.shape[1]
        assert ratio == model.output_stride

    def test_stride_is_product_of_block_strides(self) -> None:
        cfg = ModelConfig(
            n_channels=4, n_classes=10, d_model=8, n_blocks=3, input_stride=4, block_stride=2
        )
        model = EEGToTextCTC(cfg)
        assert model.output_stride == 4 * 2 * 2

    def test_single_block_uses_input_stride_only(self) -> None:
        cfg = ModelConfig(
            n_channels=4, n_classes=10, d_model=8, n_blocks=1, input_stride=8, block_stride=4
        )
        assert EEGToTextCTC(cfg).output_stride == 8


class TestLengthConversion:
    def test_lengths_match_actual_frames_exactly(self, config: ModelConfig) -> None:
        """The formula must reproduce the encoder's real frame count."""
        model = EEGToTextCTC(config).eval()
        for t in (800, 2000, 4000, 5000, 8192):
            with torch.no_grad():
                actual = model(torch.randn(1, config.n_channels, t)).shape[1]
            reported = model.input_lengths_to_output(torch.tensor([t])).item()
            assert reported == actual, f"T={t}: reported {reported}, actual {actual}"

    def test_shorter_input_yields_shorter_output(self, config: ModelConfig) -> None:
        model = EEGToTextCTC(config)
        lengths = torch.tensor([8000, 4000])
        out = model.input_lengths_to_output(lengths)
        assert out[0] > out[1]

    def test_padded_trials_map_to_at_least_one_frame(self, config: ModelConfig) -> None:
        """CTC cannot handle a zero-length prediction."""
        model = EEGToTextCTC(config)
        out = model.input_lengths_to_output(torch.tensor([1]))
        assert out.item() >= 1

    def test_converted_lengths_never_exceed_actual_frames(self, config: ModelConfig) -> None:
        """Over-reporting input_lengths to CTC yields silent shape errors."""
        model = EEGToTextCTC(config)
        for t in (1000, 3000, 5000, 7321):
            with torch.no_grad():
                frames = model(torch.randn(1, config.n_channels, t)).shape[1]
            reported = model.input_lengths_to_output(torch.tensor([t])).item()
            assert reported <= frames, f"T={t}: reported {reported} > actual {frames}"


class TestGradient:
    def test_backward_produces_finite_gradients(self, config: ModelConfig) -> None:
        model = EEGToTextCTC(config)
        out = model(torch.randn(2, config.n_channels, 2000))
        out.sum().backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads
        assert all(torch.isfinite(g).all() for g in grads)

    def test_at_least_one_parameter_receives_gradient(self, config: ModelConfig) -> None:
        model = EEGToTextCTC(config)
        model(torch.randn(1, config.n_channels, 2000)).sum().backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


class TestDeterminism:
    def test_eval_mode_is_deterministic(self, config: ModelConfig) -> None:
        """Dropout must be disabled in eval so evaluation is reproducible."""
        model = EEGToTextCTC(config).eval()
        x = torch.randn(1, config.n_channels, 3000)
        with torch.no_grad():
            a, b = model(x), model(x)
        assert torch.equal(a, b)

    def test_variable_length_inputs_are_supported(self, config: ModelConfig) -> None:
        """Trials differ in length; batching must not require fixed crops."""
        model = EEGToTextCTC(config).eval()
        with torch.no_grad():
            short = model(torch.randn(1, config.n_channels, 2000))
            long = model(torch.randn(1, config.n_channels, 8000))
        assert long.shape[1] > short.shape[1]
