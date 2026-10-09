"""CTC model: EEG encoder plus a character-level CTC head.

The encoder reduces the temporal dimension aggressively because EEG trials are
long (tens of thousands of samples at 1000 Hz) while targets are only ~6-15
characters. CTC requires ``input_length >= target_length``, so downsampling is
what makes the task feasible at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for :class:`EEGToTextCTC`."""

    n_channels: int = 128
    n_classes: int = 1995
    d_model: int = 256
    n_blocks: int = 4
    kernel_size: int = 9
    dropout: float = 0.2
    #: Temporal stride applied by the input convolution.
    input_stride: int = 8
    #: Temporal stride applied inside every residual block.
    block_stride: int = 4

    @property
    def total_stride(self) -> int:
        return self.input_stride * self.block_stride ** (self.n_blocks - 1)


class ConvBlock(nn.Module):
    """Depthwise-separable residual convolution block with temporal stride.

    When ``stride > 1`` the residual path is downsampled with a strided average
    pool so both branches share the same length (and, importantly, the same
    time alignment).
    """

    def __init__(self, d_model: int, kernel_size: int, stride: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.stride = stride
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size, stride=stride, padding=padding, groups=d_model
        )
        self.pointwise = nn.Conv1d(d_model, d_model, 1)
        self.norm = nn.GroupNorm(1, d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pointwise(self.act(self.norm(self.depthwise(x))))
        residual = x
        if self.stride > 1:
            residual = nn.functional.avg_pool1d(
                residual, kernel_size=self.stride, stride=self.stride
            )
        # Conv padding can emit one extra frame; trim to the residual length so
        # the addition is always well defined.
        if residual.shape[-1] != y.shape[-1]:
            n = min(residual.shape[-1], y.shape[-1])
            residual, y = residual[..., :n], y[..., :n]
        return self.drop(residual + y)


class EEGToTextCTC(nn.Module):
    """Maps ``(batch, channels, time)`` EEG to ``(batch, time', n_classes)`` logits."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.input_conv = nn.Conv1d(
            config.n_channels,
            config.d_model,
            config.kernel_size,
            stride=config.input_stride,
            padding=config.kernel_size // 2,
        )
        self.input_norm = nn.GroupNorm(1, config.d_model)
        self.blocks = nn.ModuleList(
            ConvBlock(
                config.d_model,
                config.kernel_size,
                config.block_stride if i > 0 else 1,
                config.dropout,
            )
            for i in range(config.n_blocks)
        )
        self.head = nn.Conv1d(config.d_model, config.n_classes, 1)

    @property
    def output_stride(self) -> int:
        """Total temporal downsampling applied to the input.

        Derived from the actual convolution strides rather than a separate
        formula, so it cannot drift from the module's real behaviour.
        """
        stride = self.input_conv.stride[0]
        for block in self.blocks:
            stride *= block.stride
        return stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C, T)`` -> ``(B, T', n_classes)``."""
        if x.ndim != 3:
            raise ValueError(f"expected (batch, channels, time), got shape {tuple(x.shape)}")
        if x.shape[1] != self.config.n_channels:
            raise ValueError(f"expected {self.config.n_channels} channels, got {x.shape[1]}")
        y = self.input_conv(x)
        y = self.input_norm(y)
        for block in self.blocks:
            y = block(y)
        logits = self.head(y)
        return logits.transpose(1, 2)

    @property
    def layer_strides(self) -> tuple[int, ...]:
        """Strides of every length-changing layer, in execution order."""
        return (
            self.input_conv.stride[0],
            *(block.stride for block in self.blocks),
        )

    def _conv_length(self, n: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        """Output length of ``conv`` for an input of ``n`` samples."""
        kernel = conv.kernel_size[0]
        stride = conv.stride[0]
        # A 1-D convolution stores padding as a single int (or a 1-tuple).
        pad = conv.padding[0] if isinstance(conv.padding, tuple) else conv.padding
        return torch.div(n + 2 * pad - kernel, stride, rounding_mode="floor") + 1

    def input_lengths_to_output(self, lengths: torch.Tensor) -> torch.Tensor:
        """Convert input sample counts to output frame counts.

        Replays the encoder's own arithmetic. Within a strided residual block
        the average-pooled residual path is the shorter branch, so it is the
        one that survives the trim: the block's output is ``floor(n / stride)``.

        This must match the module's real behaviour exactly: over-reporting to
        CTC makes it read padding frames as if they were signal.
        """
        out = self._conv_length(lengths, self.input_conv)
        for block in self.blocks:
            pooled = self._conv_length(out, block.depthwise)
            if block.stride > 1:
                pooled = torch.minimum(pooled, torch.div(out, block.stride, rounding_mode="floor"))
            out = pooled
        return torch.clamp(out, min=1)
