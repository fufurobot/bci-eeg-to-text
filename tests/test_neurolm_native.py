"""Tests for the native NeuroLM tokenizer implementation.

NeuroLM encodes EEG into discrete neural tokens with a vector-quantized
transformer, then feeds those tokens to a language model. These tests cover the
pieces needed to load released weights and to act as a NeuroProcessor.
"""

from __future__ import annotations

import pytest
import torch

from bci_eeg_to_text.neurolm_native import (
    NeuralTransformerVQ,
    PatchEmbed,
    TransformerBlock,
    VectorQuantizer,
    VQConfig,
)


@pytest.fixture
def small_config() -> VQConfig:
    return VQConfig(
        n_layer=2,
        n_head=4,
        n_embd=32,
        block_size=64,
        patch_size=50,
        codebook_size=16,
        codebook_dim=8,
    )


@pytest.fixture
def tokenizer(small_config: VQConfig) -> NeuralTransformerVQ:
    return NeuralTransformerVQ(n_channels=8, config=small_config)


class TestPatchEmbed:
    def test_patches_the_time_axis(self) -> None:
        embed = PatchEmbed(n_channels=4, patch_size=50, n_embd=16)
        out = embed(torch.randn(2, 4, 500))
        assert out.shape == (2, 10, 16)

    def test_stride_is_the_patch_size(self) -> None:
        assert PatchEmbed(4, 25, 16).stride == 25

    def test_ignores_a_trailing_partial_patch(self) -> None:
        embed = PatchEmbed(n_channels=4, patch_size=50, n_embd=16)
        assert embed(torch.randn(1, 4, 120)).shape[1] == 2


class TestTransformerBlock:
    def test_preserves_shape(self) -> None:
        block = TransformerBlock(VQConfig(n_layer=1, n_head=4, n_embd=32))
        x = torch.randn(2, 10, 32)
        assert block(x).shape == x.shape

    def test_is_causal(self) -> None:
        """A change late in the sequence must not affect earlier outputs."""
        block = TransformerBlock(VQConfig(n_layer=1, n_head=4, n_embd=32)).eval()
        x = torch.randn(1, 6, 32)
        with torch.no_grad():
            original = block(x)
            modified = x.clone()
            modified[:, -1, :] += 5.0
            changed = block(modified)
        assert torch.allclose(original[:, :-1], changed[:, :-1], atol=1e-5)
        assert not torch.allclose(original[:, -1], changed[:, -1])

    def test_rejects_indivisible_head_count(self) -> None:
        with pytest.raises(ValueError, match="must be divisible"):
            TransformerBlock(VQConfig(n_embd=30, n_head=7))


class TestVectorQuantizer:
    def test_returns_expected_shapes(self) -> None:
        quantizer = VectorQuantizer(codebook_size=8, codebook_dim=4)
        quantized, indices, loss = quantizer(torch.randn(2, 5, 4))
        assert quantized.shape == (2, 5, 4)
        assert indices.shape == (2, 5)
        assert loss.ndim == 0

    def test_indices_are_within_the_codebook(self) -> None:
        quantizer = VectorQuantizer(codebook_size=8, codebook_dim=4)
        _, indices, _ = quantizer(torch.randn(2, 20, 4) * 10)
        assert int(indices.min()) >= 0
        assert int(indices.max()) < 8

    def test_commitment_loss_is_non_negative(self) -> None:
        quantizer = VectorQuantizer(codebook_size=8, codebook_dim=4)
        _, _, loss = quantizer(torch.randn(2, 5, 4))
        assert float(loss) >= 0

    def test_straight_through_gradients_reach_the_input(self) -> None:
        """Without this the encoder would receive no learning signal."""
        quantizer = VectorQuantizer(codebook_size=8, codebook_dim=4)
        x = torch.randn(2, 5, 4, requires_grad=True)
        quantized, _, loss = quantizer(x)
        (quantized.sum() + loss).backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


class TestNeuralTransformerVQ:
    def test_forward_returns_quantized_tokens(self, tokenizer: NeuralTransformerVQ) -> None:
        quantized, indices, loss = tokenizer(torch.randn(2, 8, 500))
        assert quantized.shape == (2, 10, 8)
        assert indices.shape == (2, 10)
        assert torch.isfinite(loss)

    def test_encode_is_continuous(self, tokenizer: NeuralTransformerVQ) -> None:
        """The processor path stays continuous so CTC can train end to end."""
        features = tokenizer.encode(torch.randn(2, 8, 500))
        assert features.shape == (2, 10, 8)

    def test_encode_is_differentiable(self, tokenizer: NeuralTransformerVQ) -> None:
        features = tokenizer.encode(torch.randn(1, 8, 500))
        features.sum().backward()
        assert any(p.grad is not None for p in tokenizer.parameters())

    def test_output_stride_matches_patch_size(self, tokenizer: NeuralTransformerVQ) -> None:
        assert tokenizer.output_stride == 50
        assert tokenizer.d_model == 8

    def test_length_mapping(self, tokenizer: NeuralTransformerVQ) -> None:
        lengths = torch.tensor([500, 250])
        assert tokenizer.input_lengths_to_output(lengths).tolist() == [10, 5]

    def test_short_input_yields_at_least_one_token(self, tokenizer: NeuralTransformerVQ) -> None:
        """Zero tokens would make the downstream CTC loss undefined."""
        assert tokenizer.input_lengths_to_output(torch.tensor([10])).item() >= 1

    def test_rejects_sequence_beyond_block_size(self, tokenizer: NeuralTransformerVQ) -> None:
        # patch_size 50 * block_size 64 = 3200 samples maximum.
        with pytest.raises(ValueError, match="exceeds block_size"):
            tokenizer.encode(torch.randn(1, 8, 5000))

    def test_trainable_parameter_count_is_positive(self, tokenizer: NeuralTransformerVQ) -> None:
        assert tokenizer.trainable_parameter_count() > 0

    def test_forward_is_deterministic_in_eval(self, tokenizer: NeuralTransformerVQ) -> None:
        tokenizer.eval()
        x = torch.randn(1, 8, 500)
        with torch.no_grad():
            a, b = tokenizer.encode(x), tokenizer.encode(x)
        assert torch.equal(a, b)


class TestConfigDefaults:
    def test_defaults_match_published_settings(self) -> None:
        config = VQConfig()
        assert config.n_layer == 12
        assert config.n_embd == 400
        assert config.codebook_size == 8192

    def test_neurolm_shape_builds(self) -> None:
        """The published dimensions must at least construct."""
        model = NeuralTransformerVQ(n_channels=128, config=VQConfig(n_layer=1, block_size=8))
        assert model.output_stride == 200
