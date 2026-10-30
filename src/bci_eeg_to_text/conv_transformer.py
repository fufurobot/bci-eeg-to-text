"""Convolutional front-end plus transformer encoder for CTC EEG-to-text.

Architecture:

1. **Convolutional front-end** reduces 128 channels x 1152 samples (~18 s at
   64 Hz) to a short sequence of feature frames, using stride rather than
   pooling so temporal alignment is preserved for CTC.
2. **Transformer encoder** attends over the whole trial at once, so a prediction
   at one position is informed by the entire sentence rather than only a local
   window. Every sub-layer is pre-normed (LayerNorm) with a residual connection.
3. **CTC head** emits a distribution over characters per frame. CTC decodes the
   whole sentence in a single forward pass -- no autoregressive loop -- which is
   what makes "output the full sentence at once" true here.

Pre-norm residual blocks are the standard, more stable arrangement:

    x = x + Attention(LayerNorm(x))
    x = x + FeedForward(LayerNorm(x))
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ConvTransformerConfig:
    """Dimensions of the conv + transformer encoder."""

    n_channels: int = 128
    #: Conv front-end widths; length is the number of conv stages.
    conv_channels: tuple[int, ...] = (128, 256, 384)
    #: Temporal stride of each conv stage; their product is the output stride.
    conv_strides: tuple[int, ...] = (2, 2, 8)
    kernel_size: int = 7
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    ffn_multiplier: float = 4.0
    dropout: float = 0.1
    #: Maximum frames the positional encoding can address.
    max_frames: int = 512

    def __post_init__(self) -> None:
        if len(self.conv_channels) != len(self.conv_strides):
            raise ValueError(
                f"conv_channels ({len(self.conv_channels)}) and conv_strides "
                f"({len(self.conv_strides)}) must have equal length"
            )
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )

    @property
    def output_stride(self) -> int:
        stride = 1
        for s in self.conv_strides:
            stride *= s
        return stride


class ConvStage(nn.Module):
    """A strided convolution followed by GroupNorm, GELU and a residual path.

    GroupNorm with one group normalises across the whole channel axis, which is
    what a conv feature map wants; the transformer blocks use LayerNorm instead.

    When the stride exceeds 1 the residual branch is downsampled and re-projected
    with a 1x1 convolution so the shapes match; that keeps the residual exact
    rather than falling back to a zero-padding shortcut.
    """

    def __init__(self, c_in: int, c_out: int, kernel_size: int, stride: int, dropout: float):
        super().__init__()
        padding = kernel_size // 2
        self.stride = stride
        self.conv = nn.Conv1d(c_in, c_out, kernel_size, stride=stride, padding=padding)
        self.norm = nn.GroupNorm(1, c_out)  # applied before the channel mix
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        # Residual branch: identity when shapes already match, else a strided
        # convolution. A strided conv is used rather than average pooling
        # because pooling computes floor(n/s) while the main conv computes
        # floor((n-1)/s)+1; the two disagree for some n, and the shorter path
        # would silently truncate the output and break CTC length accounting.
        if stride == 1 and c_in == c_out:
            self.shortcut: nn.Module = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.AvgPool1d(stride, ceil_mode=True) if stride > 1 else nn.Identity(),
                nn.Conv1d(c_in, c_out, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.drop(self.act(self.norm(self.conv(x))))
        residual = self.shortcut(x)
        # Ceil-mode pooling can overshoot by one frame; pad the conv output up
        # rather than truncating, so the length is never reduced below the
        # convolution's own output.
        if residual.shape[-1] < y.shape[-1]:
            residual = F.pad(residual, (0, y.shape[-1] - residual.shape[-1]))
        elif residual.shape[-1] > y.shape[-1]:
            residual = residual[..., : y.shape[-1]]
        return y + residual


class PreNormEncoderLayer(nn.Module):
    """Transformer layer: pre-norm residual attention, then pre-norm residual FFN."""

    def __init__(self, d_model: int, n_heads: int, ffn_multiplier: float, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * ffn_multiplier)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Self-attention over the full sequence, then a position-wise FFN.

        ``key_padding_mask`` marks padded frames so attention does not read them.
        Attention is bidirectional: every frame sees the whole trial, which is
        what lets one forward pass commit to the entire sentence.
        """
        b, t, d = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_mask = None
        if key_padding_mask is not None:
            # (B, T) bool where True means "ignore this key".
            attn_mask = key_padding_mask[:, None, None, :].expand(b, self.n_heads, t, t)
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.drop.p if self.training else 0.0
        )
        attended = attended.transpose(1, 2).reshape(b, t, d)
        x = x + self.drop(self.proj(attended))
        return x + self.drop(self.ffn(self.norm2(x)))


class ConvTransformerEncoder(nn.Module):
    """Convolutional front-end + transformer encoder, returning ``(B, T, d_model)``."""

    def __init__(self, config: ConvTransformerConfig) -> None:
        super().__init__()
        self.config = config
        stages: list[nn.Module] = []
        c_in = config.n_channels
        for c_out, stride in zip(config.conv_channels, config.conv_strides, strict=True):
            stages.append(ConvStage(c_in, c_out, config.kernel_size, stride, config.dropout))
            c_in = c_out
        self.conv = nn.Sequential(*stages)
        self.to_model = (
            nn.Identity() if c_in == config.d_model else nn.Conv1d(c_in, config.d_model, 1)
        )
        self.pos = nn.Parameter(torch.zeros(1, config.max_frames, config.d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.layers = nn.ModuleList(
            PreNormEncoderLayer(
                config.d_model, config.n_heads, config.ffn_multiplier, config.dropout
            )
            for _ in range(config.n_layers)
        )
        self.norm = nn.LayerNorm(config.d_model)

    @property
    def output_stride(self) -> int:
        return self.config.output_stride

    @property
    def d_model(self) -> int:
        return self.config.d_model

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """``(B, C, T)`` EEG -> ``(B, T', d_model)`` contextual features."""
        if x.ndim != 3:
            raise ValueError(f"expected (batch, channels, time), got {tuple(x.shape)}")
        y = self.to_model(self.conv(x)).transpose(1, 2)  # (B, T', d)
        t = y.shape[1]
        if t > self.config.max_frames:
            raise ValueError(f"sequence of {t} frames exceeds max_frames {self.config.max_frames}")
        y = y + self.pos[:, :t]

        mask = None
        if lengths is not None:
            valid = self.input_lengths_to_output(lengths)
            idx = torch.arange(t, device=y.device).unsqueeze(0)
            mask = idx >= valid.unsqueeze(1)  # True where padded
        for layer in self.layers:
            y = layer(y, mask)
        return self.norm(y)

    def input_lengths_to_output(self, lengths: torch.Tensor) -> torch.Tensor:
        """Replay the convolution arithmetic exactly.

        Each stage is ``floor((n + 2p - k) / s) + 1`` with ``p = k // 2``, which
        for odd ``k`` is ``floor((n - 1) / s) + 1``. Reporting lengths that
        disagree with reality makes CTC consume padding as if it were signal.
        """
        frames = lengths
        kernel = self.config.kernel_size
        for stride in self.config.conv_strides:
            frames = (
                torch.div(frames + 2 * (kernel // 2) - kernel, stride, rounding_mode="floor") + 1
            )
        return frames.to(torch.long)


def ctc_length_ratio(stride: int) -> float:
    """Frames per sample, for planning sequence lengths."""
    return 1.0 / stride


def estimate_frames(n_samples: int, stride: int) -> int:
    """Frames produced for ``n_samples`` samples."""
    if n_samples <= 0 or stride <= 0:
        return 0
    return int(math.floor((n_samples - 1) / stride)) + 1
