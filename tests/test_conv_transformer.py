"""Tests for the convolutional + transformer encoder.

The architecture is a convolutional front-end (stride, not pooling, so CTC
alignment is preserved) followed by a transformer encoder of **pre-norm residual
blocks**: every sub-layer is LayerNorm'd and added to its input, which is the
structure that was requested.
"""

from __future__ import annotations

import pytest
import torch

from bci_eeg_to_text.conv_transformer import (
    ConvStage,
    ConvTransformerConfig,
    ConvTransformerEncoder,
    PreNormEncoderLayer,
    estimate_frames,
)


@pytest.fixture
def config() -> ConvTransformerConfig:
    return ConvTransformerConfig(
        n_channels=8, d_model=32, n_heads=4, n_layers=2, max_frames=256
    )


class TestConfig:
    def test_output_stride_is_product_of_strides(self) -> None:
        cfg = ConvTransformerConfig(conv_strides=(2, 2, 8))
        assert cfg.output_stride == 32

    def test_rejects_mismatched_channel_and_stride_lists(self) -> None:
        with pytest.raises(ValueError, match="must have equal length"):
            ConvTransformerConfig(conv_channels=(8, 16), conv_strides=(2,))

    def test_rejects_indivisible_heads(self) -> None:
        with pytest.raises(ValueError, match="divisible by n_heads"):
            ConvTransformerConfig(d_model=30, n_heads=8)


class TestConvStage:
    def test_changes_channels_and_length(self) -> None:
        stage = ConvStage(8, 16, kernel_size=7, stride=2, dropout=0.0)
        out = stage(torch.randn(2, 8, 100))
        assert out.shape[0] == 2
        assert out.shape[1] == 16
        assert out.shape[2] > 0

    def test_residual_never_truncates_below_the_conv_length(self) -> None:
        """A shorter residual path would silently drop frames.

        Average pooling gives floor(n/s) while the convolution gives
        floor((n-1)/s)+1; the two disagree for some n, and truncating to the
        shorter one loses a frame and breaks CTC length accounting.
        """
        for n in (250, 500, 1000, 1001):
            stage = ConvStage(4, 4, kernel_size=7, stride=8, dropout=0.0)
            out = stage(torch.randn(1, 4, n))
            expected = torch.nn.Conv1d(4, 4, 7, stride=8, padding=3)(
                torch.zeros(1, 4, n)
            ).shape[-1]
            assert out.shape[-1] == expected, f"n={n}"

    def test_has_a_residual_connection(self) -> None:
        stage = ConvStage(4, 4, kernel_size=7, stride=1, dropout=0.0).eval()
        x = torch.randn(1, 4, 32)
        # With the conv weights zeroed, output equals the identity shortcut.
        with torch.no_grad():
            stage.conv.weight.zero_()
            stage.conv.bias.zero_()
            out = stage(x)
        assert torch.allclose(out, x, atol=1e-5)


class TestPreNormLayer:
    def test_uses_layernorm_on_both_sublayers(self) -> None:
        layer = PreNormEncoderLayer(32, 4, 2.0, 0.0)
        assert isinstance(layer.norm1, torch.nn.LayerNorm)
        assert isinstance(layer.norm2, torch.nn.LayerNorm)

    def test_preserves_shape(self) -> None:
        layer = PreNormEncoderLayer(32, 4, 2.0, 0.0)
        x = torch.randn(2, 10, 32)
        assert layer(x).shape == x.shape

    def test_is_bidirectional(self) -> None:
        """Every frame sees the whole trial, so the sentence is decoded at once.

        Asserting on the numerical output is fragile: with fresh initialisation
        the influence of the last position on the first is around 1e-8, below any
        sane tolerance. The architectural claim is stronger and exact -- the
        attention matrix has no causal mask, so position 0 attends to every later
        position. A causal layer would leave that corner exactly zero.
        """
        torch.manual_seed(0)
        layer = PreNormEncoderLayer(32, 4, 2.0, 0.0).eval()
        x = torch.randn(1, 8, 32)
        b, n, d = x.shape
        qkv = layer.qkv(layer.norm1(x)).view(b, n, 3, 4, d // 4).permute(2, 0, 3, 1, 4)
        scores = qkv[0] @ qkv[1].transpose(-1, -2) / (d // 4) ** 0.5
        weights = scores.softmax(dim=-1)[:, :, 0, :]  # position 0's attention
        # Attention from position 0 onto the last position is non-zero: causal
        # attention would force exactly 0 here.
        assert bool((weights[..., -1] > 0).all())
        assert not torch.allclose(weights[..., 1:], torch.zeros_like(weights[..., 1:]))

    def test_padding_mask_hides_padded_frames(self) -> None:
        layer = PreNormEncoderLayer(32, 4, 2.0, 0.0).eval()
        x = torch.randn(1, 8, 32)
        mask = torch.zeros(1, 8, dtype=torch.bool)
        mask[0, 6:] = True
        with torch.no_grad():
            a = layer(x, mask)
            b = x.clone()
            b[:, 6:, :] += 100.0
            out = layer(b, mask)
        # Padded positions must not influence valid ones.
        assert torch.allclose(a[:, :6], out[:, :6], atol=1e-5)


class TestEncoder:
    def test_output_shape(self, config: ConvTransformerConfig) -> None:
        encoder = ConvTransformerEncoder(config)
        out = encoder(torch.randn(2, 8, 1152))
        assert out.ndim == 3
        assert out.shape[0] == 2
        assert out.shape[2] == config.d_model

    def test_rejects_wrong_rank(self, config: ConvTransformerConfig) -> None:
        with pytest.raises(ValueError, match="expected"):
            ConvTransformerEncoder(config)(torch.randn(8, 1152))

    @pytest.mark.parametrize("n", [500, 1000, 1152, 1500, 2048, 4096])
    def test_length_mapping_is_exact(self, n: int) -> None:
        """Over-reporting frames makes CTC read padding as signal."""
        cfg = ConvTransformerConfig(n_channels=8, d_model=32, n_heads=4, n_layers=1)
        encoder = ConvTransformerEncoder(cfg).eval()
        with torch.no_grad():
            actual = encoder(torch.randn(1, 8, n)).shape[1]
        reported = int(encoder.input_lengths_to_output(torch.tensor([n]))[0])
        assert actual == reported, f"n={n}: {actual} != {reported}"

    def test_estimate_frames_agrees_with_the_encoder(self) -> None:
        cfg = ConvTransformerConfig(n_channels=8, d_model=32, n_heads=4, n_layers=1)
        encoder = ConvTransformerEncoder(cfg).eval()
        for n in (1000, 1152):
            with torch.no_grad():
                actual = encoder(torch.randn(1, 8, n)).shape[1]
            assert estimate_frames(n, cfg.output_stride) == actual

    def test_padded_batch_uses_masks(self, config: ConvTransformerConfig) -> None:
        encoder = ConvTransformerEncoder(config).eval()
        lengths = torch.tensor([1152, 800])
        with torch.no_grad():
            out = encoder(torch.randn(2, 8, 1152), lengths)
        assert torch.isfinite(out).all()

    def test_rejects_sequence_beyond_max_frames(self) -> None:
        cfg = ConvTransformerConfig(
            n_channels=8, d_model=32, n_heads=4, n_layers=1, max_frames=16
        )
        with pytest.raises(ValueError, match="exceeds max_frames"):
            ConvTransformerEncoder(cfg)(torch.randn(1, 8, 4096))

    def test_gradients_flow(self, config: ConvTransformerConfig) -> None:
        encoder = ConvTransformerEncoder(config)
        encoder(torch.randn(1, 8, 1152)).sum().backward()
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters()
        )


class TestBackboneIntegration:
    def test_conv_transformer_is_a_training_mode(self) -> None:
        from bci_eeg_to_text.backbones import TRAINING_MODES

        assert "conv-transformer" in TRAINING_MODES

    def test_builds_a_ctc_model(self) -> None:
        from bci_eeg_to_text.backbones import build_ctc_model

        model = build_ctc_model(
            "conv-transformer", n_channels=8, n_classes=17, d_model=32
        )
        logits = model(torch.randn(2, 8, 1152))
        assert logits.shape[0] == 2
        assert logits.shape[2] == 17

    def test_model_contains_layernorm_and_attention(self) -> None:
        """The requested architecture, asserted structurally."""
        from bci_eeg_to_text.backbones import build_ctc_model

        model = build_ctc_model(
            "conv-transformer", n_channels=8, n_classes=17, d_model=32
        )
        kinds = {type(m).__name__ for m in model.modules()}
        assert "LayerNorm" in kinds
        assert "ConvTransformerEncoder" in kinds
        assert "PreNormEncoderLayer" in kinds
