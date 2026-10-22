"""HuggingFace-native conv-transformer encoder for CTC EEG-to-text.

A trained checkpoint of this package publishes as an ordinary HuggingFace folder
and loads through the stock entry points::

    AutoConfig.from_pretrained(folder, trust_remote_code=True)
    AutoTokenizer.from_pretrained(folder, trust_remote_code=True)
    AutoProcessor.from_pretrained(folder, trust_remote_code=True)
    AutoModel.from_pretrained(folder, trust_remote_code=True)

The architecture is the one we train: a convolutional front-end that reduces a
long raw trial to a short sequence, followed by pre-norm residual transformer
blocks (LayerNorm on both sub-layers, bidirectional attention) and a per-frame
character head. CTC decodes the whole sentence in a single forward pass.

The loader also accepts **any** HuggingFace folder via
:func:`load_any_backbone`, which resolves ``AutoModel`` with
``trust_remote_code=True`` so an external architecture can be dropped in.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

try:  # transformers is optional at import time, required for the HF entry points
    from transformers import AutoModel, PretrainedConfig, PreTrainedModel
    from transformers.modeling_outputs import ModelOutput

    _HF = True
except ImportError:  # pragma: no cover - exercised only without transformers
    _HF = False

    class PretrainedConfig:  # type: ignore[no-redef]
        """Minimal stand-in so the module imports without transformers."""

        model_type = "eeg-ctc"

        def __init__(self, **kwargs) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

        def save_pretrained(self, directory) -> None:
            import json
            from pathlib import Path

            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            payload = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
            (path / "config.json").write_text(json.dumps(payload, indent=2))

        @classmethod
        def from_pretrained(cls, directory, **kwargs):
            import json
            from pathlib import Path

            return cls(**json.loads((Path(directory) / "config.json").read_text()))

    class PreTrainedModel:  # type: ignore[no-redef]
        pass

    class AutoModel:  # type: ignore[no-redef]
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise BackboneLoadError("transformers is not installed")

    class ModelOutput:  # type: ignore[no-redef]
        def __init__(self, logits=None, **kwargs) -> None:
            self.logits = logits


if _HF:
    from dataclasses import dataclass as _dc

    @_dc
    class EEGCTCOutput(ModelOutput):
        """Model outputs: ``logits`` is ``(batch, frames, n_classes)``."""

        logits: torch.Tensor | None = None
else:  # pragma: no cover
    EEGCTCOutput = dict  # type: ignore[assignment,misc]


class BackboneLoadError(RuntimeError):
    """Raised when a HuggingFace folder cannot be loaded as a backbone."""


class EEGCTCConfig(PretrainedConfig):
    """Configuration for :class:`EEGCTCModel`.

    ``model_type`` is what lets ``AutoConfig`` find this class through the
    vendored custom code, so it must stay stable across releases.
    """

    model_type = "eeg-ctc"

    def __init__(
        self,
        n_channels: int = 128,
        n_classes: int = 1002,
        conv_channels: tuple[int, ...] | list[int] = (128, 256, 384),
        conv_strides: tuple[int, ...] | list[int] = (2, 2, 8),
        kernel_size: int = 7,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        ffn_multiplier: float = 4.0,
        dropout: float = 0.1,
        max_frames: int = 512,
        **kwargs,
    ) -> None:
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.conv_channels = list(conv_channels)
        self.conv_strides = list(conv_strides)
        self.kernel_size = kernel_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ffn_multiplier = ffn_multiplier
        self.dropout = dropout
        self.max_frames = max_frames
        super().__init__(**kwargs)

    @property
    def output_stride(self) -> int:
        """Raw samples consumed per output frame."""
        stride = 1
        for s in self.conv_strides:
            stride *= s
        return stride


class _ConvStage(torch.nn.Module):
    """Strided convolution with a matching residual path, then norm and GELU."""

    def __init__(self, c_in: int, c_out: int, k: int, stride: int, dropout: float) -> None:
        super().__init__()
        self.stride = stride
        self.conv = torch.nn.Conv1d(c_in, c_out, k, stride=stride, padding=k // 2)
        self.norm = torch.nn.GroupNorm(1, c_out)
        self.act = torch.nn.GELU()
        self.drop = torch.nn.Dropout(dropout)
        if stride == 1 and c_in == c_out:
            self.shortcut: torch.nn.Module = torch.nn.Identity()
        else:
            self.shortcut = torch.nn.Sequential(
                torch.nn.AvgPool1d(stride, ceil_mode=True) if stride > 1 else torch.nn.Identity(),
                torch.nn.Conv1d(c_in, c_out, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.drop(self.act(self.norm(self.conv(x))))
        residual = self.shortcut(x)
        # Never truncate below the convolution's own output: pooling and
        # striding round differently and the shorter path would drop a frame.
        if residual.shape[-1] < y.shape[-1]:
            residual = F.pad(residual, (0, y.shape[-1] - residual.shape[-1]))
        elif residual.shape[-1] > y.shape[-1]:
            residual = residual[..., : y.shape[-1]]
        return y + residual


class _EncoderLayer(torch.nn.Module):
    """Pre-norm residual transformer layer: attention, then feed-forward."""

    def __init__(self, d_model: int, n_heads: int, ffn_mult: float, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.qkv = torch.nn.Linear(d_model, 3 * d_model)
        self.proj = torch.nn.Linear(d_model, d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        hidden = int(d_model * ffn_mult)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, d_model),
        )
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv(self.norm1(x)).view(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn_mask = None
        if mask is not None:
            attn_mask = mask[:, None, None, :].expand(b, self.n_heads, t, t)
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.drop.p if self.training else 0.0
        )
        attended = attended.transpose(1, 2).reshape(b, t, d)
        x = x + self.drop(self.proj(attended))
        return x + self.drop(self.ffn(self.norm2(x)))


class _Encoder(torch.nn.Module):
    """Convolutional front-end plus transformer blocks."""

    def __init__(self, config: EEGCTCConfig) -> None:
        super().__init__()
        self.config = config
        stages: list[torch.nn.Module] = []
        c_in = config.n_channels
        for c_out, stride in zip(config.conv_channels, config.conv_strides, strict=True):
            stages.append(_ConvStage(c_in, c_out, config.kernel_size, stride, config.dropout))
            c_in = c_out
        self.conv = torch.nn.Sequential(*stages)
        self.to_model = (
            torch.nn.Identity()
            if c_in == config.d_model
            else torch.nn.Conv1d(c_in, config.d_model, 1)
        )
        self.pos = torch.nn.Parameter(torch.zeros(1, config.max_frames, config.d_model))
        torch.nn.init.trunc_normal_(self.pos, std=0.02)
        self.layers = torch.nn.ModuleList(
            _EncoderLayer(config.d_model, config.n_heads, config.ffn_multiplier, config.dropout)
            for _ in range(config.n_layers)
        )
        self.norm = torch.nn.LayerNorm(config.d_model)

    def lengths_to_frames(self, lengths: torch.Tensor) -> torch.Tensor:
        """Replay the convolution arithmetic exactly for CTC.

        ``floor((n + 2p - k) / s) + 1`` per stage; over-reporting frames makes
        CTC read padding as signal.
        """
        frames = lengths
        for stride in self.config.conv_strides:
            frames = (
                torch.div(
                    frames + 2 * (self.config.kernel_size // 2) - self.config.kernel_size,
                    stride,
                    rounding_mode="floor",
                )
                + 1
            )
        return frames.to(torch.long)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected (batch, channels, time), got {tuple(x.shape)}")
        y = self.to_model(self.conv(x)).transpose(1, 2)
        t = y.shape[1]
        if t > self.config.max_frames:
            raise ValueError(f"sequence of {t} frames exceeds max_frames {self.config.max_frames}")
        y = y + self.pos[:, :t]
        mask = None
        if lengths is not None:
            valid = self.lengths_to_frames(lengths)
            idx = torch.arange(t, device=y.device).unsqueeze(0)
            mask = idx >= valid.unsqueeze(1)
        for layer in self.layers:
            y = layer(y, mask)
        return self.norm(y)


class EEGCTCModel(PreTrainedModel):
    """CTC EEG-to-text model, saveable and auto-loadable as an HF folder."""

    config_class = EEGCTCConfig
    base_model_prefix = "eeg_ctc"
    _keys_to_ignore_on_load_missing: list[str] = []

    def __init__(self, config: EEGCTCConfig) -> None:
        super().__init__(config)
        if _HF:
            super().__init__(config)
        self.config = config
        self.encoder = _Encoder(config)
        self.head = torch.nn.Linear(config.d_model, config.n_classes)

    def input_lengths_to_output(self, lengths: torch.Tensor) -> torch.Tensor:
        return self.encoder.lengths_to_frames(lengths)

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        input_lengths: torch.Tensor | None = None,
        **kwargs,
    ):
        """``pixel_values`` is ``(B, channels, time)`` raw EEG; returns logits."""
        if pixel_values is None:
            raise ValueError("pixel_values (EEG) is required")
        features = self.encoder(pixel_values, input_lengths)
        logits = self.head(features)
        if _HF:
            return EEGCTCOutput(logits=logits)
        return {"logits": logits}

    def save_pretrained(self, directory, **kwargs) -> None:
        """Write a self-contained folder: config, weights and vendored code."""
        from bci_eeg_to_text.hf_release import write_custom_code

        super().save_pretrained(directory, **kwargs)
        write_custom_code(directory)

    @classmethod
    def from_pretrained(cls, directory, **kwargs):
        return super().from_pretrained(directory, **kwargs)


def load_any_backbone(
    name_or_path: str,
    *,
    n_channels: int,
    d_model: int,
    trust_remote_code: bool = True,
):
    """Load **any** HuggingFace folder as an EEG feature extractor.

    ``trust_remote_code=True`` is the default because architectures with custom
    code (including our own release folder) need it; the client caches the code
    and should review it once.

    Returns a callable mapping ``(B, C, T)`` to ``(B, T', d_model)``. A model
    that emits per-frame logits is used as-is; an encoder-only model returns its
    last hidden state.
    """
    try:
        model = AutoModel.from_pretrained(name_or_path, trust_remote_code=trust_remote_code)
    except Exception as exc:  # noqa: BLE001 - surfaced with guidance below
        raise BackboneLoadError(
            f"could not load HuggingFace folder {name_or_path!r}: {exc}. "
            "Check the path/repo id and that the folder contains config.json."
        ) from exc

    # A loaded model is always used for inference here; tolerate duck-typed
    # objects that expose only a forward call.
    if hasattr(model, "eval"):
        model.eval()

    class _Wrapper(torch.nn.Module):
        """Adapts EEG to whatever input shape the loaded model expects.

        A general HuggingFace model (BERT, say) was not written for EEG, so its
        inputs are words or audio. Rather than fail, the wrapper projects the
        channel axis onto whatever width the model accepts, which makes "any
        folder AutoModel can load" actually usable as a feature extractor.
        """

        def __init__(self) -> None:
            super().__init__()
            self.inner = model
            self.adapter: torch.nn.Module | None = None

        def _prepare(self, x: torch.Tensor) -> torch.Tensor:
            if "pixel_values" in _params(self.inner):
                return x
            config = getattr(self.inner, "config", None)
            width = getattr(config, "hidden_size", None) or getattr(config, "d_model", None)
            if width is None or x.shape[1] == width:
                return x
            if self.adapter is None:
                self.adapter = torch.nn.Linear(x.shape[1], width)
            # (B, C, T) -> (B, T, C) -> project channels -> back to (B, C', T)
            projected = self.adapter(x.transpose(1, 2)).transpose(1, 2)
            return projected

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            prepared = self._prepare(x)
            params = _params(self.inner)
            try:
                if "pixel_values" in params:
                    out = self.inner(pixel_values=prepared)
                elif "inputs_embeds" in params:
                    # Text models take embeddings, not waveform values; the
                    # channel projection above stands in for the embedding layer.
                    config = getattr(self.inner, "config", None)
                    limit = getattr(config, "max_position_embeddings", None)
                    seq = prepared.transpose(1, 2)
                    if limit:
                        seq = _pool_to(seq, int(limit))
                    out = self.inner(inputs_embeds=seq)
                else:
                    out = self.inner(prepared)
            except (ValueError, TypeError, RuntimeError):
                # Non-sequence models (e.g. image or pooled encoders) reject
                # this shape; report that clearly instead of a bare unpack error.
                raise BackboneLoadError(
                    f"{type(self.inner).__name__} cannot accept EEG input of shape "
                    f"{tuple(x.shape)}; use an encoder that takes sequences"
                ) from None
            if hasattr(out, "logits"):
                return out.logits
            if hasattr(out, "last_hidden_state"):
                return out.last_hidden_state
            if torch.is_tensor(out):
                return out
            first = out[0] if isinstance(out, (tuple, list)) else out
            return first

    return _Wrapper()


def _pool_to(x: torch.Tensor, limit: int) -> torch.Tensor:
    """Adaptive-average-pool the time axis to ``limit`` if it is longer.

    Text encoders cap sequence length (BERT: 512), while a trial is thousands of
    samples; pooling keeps the whole trial represented instead of truncating it.
    """
    if x.shape[1] <= limit:
        return x
    pooled = F.adaptive_avg_pool1d(x.transpose(1, 2), limit)
    return pooled.transpose(1, 2)


def _params(module) -> set[str]:
    import inspect

    try:
        return set(inspect.signature(module.forward).parameters)
    except (TypeError, ValueError):  # pragma: no cover
        return set()


def estimate_frames(n_samples: int, stride: int) -> int:
    """Frames a stride-``stride`` encoder produces for ``n_samples``."""
    if n_samples <= 0 or stride <= 0:
        return 0
    return int(math.floor((n_samples - 1) / stride)) + 1
