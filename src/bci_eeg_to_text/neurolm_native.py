"""Native implementation of NeuroLM's neural tokenizer.

NeuroLM's EEG tokenizer is a vector-quantized transformer that predicts
temporal-frequency content. This module reimplements the parts needed to load
released weights and to act as a NeuroProcessor, without importing the upstream
research code (which vendors nanoGPT and pins old dependencies).

The encoder follows the published configuration: ``n_layer`` transformer blocks
over patch embeddings, followed by a vector-quantization bottleneck.

Everything here is deliberately small and dependency-light so it can be tested
on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VQConfig:
    """Tokenizer dimensions, defaulting to NeuroLM's published settings."""

    n_layer: int = 12
    n_head: int = 10
    n_embd: int = 400
    block_size: int = 1024
    bias: bool = False
    dropout: float = 0.0
    #: Size of the discrete neural codebook.
    codebook_size: int = 8192
    #: Embedding dimension of each codebook entry.
    codebook_dim: int = 64
    #: Number of raw EEG samples per patch.
    patch_size: int = 200


class PatchEmbed(nn.Module):
    """Split a long EEG signal into patches and embed each one.

    NeuroLM consumes 200-sample patches (1 s at its 200 Hz working rate), so a
    30 s trial becomes 30 tokens rather than 6000 samples.
    """

    def __init__(self, n_channels: int, patch_size: int, n_embd: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(n_channels, n_embd, kernel_size=patch_size, stride=patch_size)

    @property
    def stride(self) -> int:
        return self.patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C, T)`` -> ``(B, T // patch_size, n_embd)``."""
        return self.proj(x).transpose(1, 2)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with causal self-attention."""

    def __init__(self, config: VQConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError(
                f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"
            )
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, _ = x.shape
        y = self.ln_1(x)
        q, k, v = self.attn(y).split(self.n_embd, dim=2)
        head_dim = self.n_embd // self.n_head
        q = q.view(batch, time, self.n_head, head_dim).transpose(1, 2)
        k = k.view(batch, time, self.n_head, head_dim).transpose(1, 2)
        v = v.view(batch, time, self.n_head, head_dim).transpose(1, 2)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attended = attended.transpose(1, 2).contiguous().view(batch, time, self.n_embd)
        x = x + self.proj(attended)
        return x + self.mlp(self.ln_2(x))


class VectorQuantizer(nn.Module):
    """Straight-through vector quantization with an EMA-free codebook.

    Used only when discrete neural tokens are wanted (NeuroLM's own recipe).
    The CTC pipeline keeps features continuous, so this is optional.
    """

    def __init__(self, codebook_size: int, codebook_dim: int) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.embed = nn.Embedding(codebook_size, codebook_dim)
        self.embed.weight.data.uniform_(-1 / codebook_size, 1 / codebook_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize ``(B, T, D)``; returns ``(quantized, indices, commitment_loss)``."""
        flat = x.reshape(-1, x.shape[-1])
        distances = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embed.weight.t()
            + self.embed.weight.pow(2).sum(1)
        )
        indices = distances.argmin(1)
        quantized = self.embed(indices).view_as(x)
        commitment = F.mse_loss(quantized.detach(), x)
        # Straight-through estimator: gradients pass through unchanged.
        quantized = x + (quantized - x).detach()
        return quantized, indices.view(x.shape[0], x.shape[1]), commitment


class NeuralTransformerVQ(nn.Module):
    """NeuroLM's neural tokenizer: patch embed, transformer, VQ bottleneck."""

    def __init__(
        self,
        n_channels: int = 128,
        d_model: int = 400,
        config: VQConfig | None = None,
    ) -> None:
        super().__init__()
        config = config or VQConfig(n_embd=d_model)
        self.config = config
        self.patch_embed = PatchEmbed(n_channels, config.patch_size, config.n_embd)
        self.pos_embed = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        # Project to the codebook dimension before quantizing.
        self.to_codebook = nn.Linear(config.n_embd, config.codebook_dim)
        self.quantizer = VectorQuantizer(config.codebook_size, config.codebook_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def output_stride(self) -> int:
        """Raw samples consumed per output token."""
        return self.patch_embed.stride

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C, T)`` EEG -> continuous ``(B, T', codebook_dim)`` features."""
        y = self.patch_embed(x)
        time = y.shape[1]
        if time > self.config.block_size:
            raise ValueError(
                f"sequence of {time} patches exceeds block_size "
                f"{self.config.block_size}; shorten the trial or raise block_size"
            )
        positions = torch.arange(time, device=x.device)
        y = y + self.pos_embed(positions).unsqueeze(0)
        for block in self.blocks:
            y = block(y)
        return self.to_codebook(self.ln_f(y))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode and quantize; returns ``(quantized, indices, commitment_loss)``."""
        return self.quantizer(self.encode(x))

    def input_lengths_to_output(self, lengths: torch.Tensor) -> torch.Tensor:
        stride = self.output_stride
        return torch.div(lengths, stride, rounding_mode="floor").clamp(min=1)

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def d_model(self) -> int:
        """Feature width produced by :meth:`encode`."""
        return self.config.codebook_dim
