"""Interchangeable encoders for CTC EEG-to-text training.

All backbones satisfy one contract: given ``(batch, n_channels, time)`` EEG they
return ``(batch, time', d_model)`` features, and they report ``output_stride``
so CTC input lengths can be computed exactly. Everything downstream (loss,
decoding, metrics, metrics-based model selection) is shared.

Four families are provided:

``conv``
    The strided convolutional encoder from :mod:`bci_eeg_to_text.model`. Trained
    from scratch; the baseline.

``bert-ctc``
    A pretrained HuggingFace BERT encoder with a CTC head. EEG is projected into
    BERT's hidden space (an input adapter) and passed as ``inputs_embeds``. The
    backbone may be frozen, which is the point: pretrained language
    representations do the heavy lifting and only the adapter and head train.

``neurolm-sft``
    The full NeuroLM model (frozen VQ tokenizer + enlarged GPT-2), supervised
    fine-tuning on our pairs.

``neurolm-adapter``
    A HuggingFace causal LM plus a NeuroProcessor, with a trainable adapter
    between them. The NeuroProcessor can be loaded pretrained or trained from
    scratch.

NeuroLM background: EEG is encoded by a frozen vector-quantized tokenizer (the
"NeuroProcessor") into discrete neural tokens, which are fed to a GPT-2 decoder
whose embedding matrix and LM head were enlarged to hold an EEG vocabulary.
Its ``encode_transform_layer`` is precisely the adapter between the two.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn

from bci_eeg_to_text.model import EEGToTextCTC, ModelConfig

#: Training modes, in the order they are documented in the README.
TRAINING_MODES = (
    "scratch",
    "conv-transformer",
    "bert-ctc",
    "neurolm-sft",
    "neurolm-adapter",
)


class BackboneError(RuntimeError):
    """Raised when a backbone cannot be constructed or used."""


@runtime_checkable
class Backbone(Protocol):
    """Contract every encoder must satisfy."""

    @property
    def output_stride(self) -> int:
        """Total temporal downsampling applied to the input."""

    @property
    def d_model(self) -> int:
        """Feature width returned by :meth:`forward`."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C, T)`` EEG -> ``(B, T', d_model)`` features."""

    def input_lengths_to_output(self, lengths: torch.Tensor) -> torch.Tensor:
        """Map input sample counts to output frame counts."""

    def trainable_parameter_count(self) -> int:
        """Number of parameters that will receive gradients."""


@dataclass
class BackboneConfig:
    """Settings shared by all backbones."""

    n_channels: int = 128
    d_model: int = 256
    #: Pretrained checkpoint name or local path, when the family needs one.
    pretrained: str | None = None
    #: Freeze the pretrained encoder, training only adapters and heads.
    freeze_backbone: bool = False
    dropout: float = 0.1
    #: Extra family-specific options.
    extra: dict[str, object] = field(default_factory=dict)


class ConvBackbone(nn.Module):
    """Strided convolutional encoder trained from scratch."""

    family = "conv"

    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        self._model = EEGToTextCTC(
            ModelConfig(
                n_channels=config.n_channels,
                n_classes=config.d_model,  # unused for features
                d_model=config.d_model,
                dropout=config.dropout,
            )
        )

    @property
    def output_stride(self) -> int:
        return self._model.output_stride

    @property
    def d_model(self) -> int:
        return self._model.config.d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected (batch, channels, time), got {tuple(x.shape)}")
        model = self._model
        y = model.input_norm(model.input_conv(x))
        for block in model.blocks:
            y = block(y)
        return y.transpose(1, 2)

    def input_lengths_to_output(self, lengths: torch.Tensor) -> torch.Tensor:
        return self._model.input_lengths_to_output(lengths).to(torch.long)

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class _LengthPreservingMixin:
    """Length bookkeeping for backbones that do not subsample further."""

    def input_lengths_to_output(self, lengths: torch.Tensor) -> torch.Tensor:
        stride = self.output_stride
        if stride == 1:
            return lengths.clone()
        return self._base_length_map(lengths, stride)


def make_bert_backbone(config: BackboneConfig) -> type[nn.Module]:
    """Build the (optional) BERT-CTC backbone class.

    Imported lazily so the core pipeline keeps working without ``transformers``
    installed; the failure message says exactly what to install.
    """
    try:
        from transformers import BertConfig, BertModel
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise BackboneError(
            "the 'bert-ctc' mode requires the transformers package; "
            "install it with `uv add transformers`"
        ) from exc

    default_name = config.pretrained or "bert-base-chinese"

    class BertCTCBackbone(nn.Module):
        """Pretrained BERT encoder fed projected EEG embeddings.

        EEG has no token ids, so we project it to BERT's hidden size and pass it
        through ``inputs_embeds``. That keeps the whole pretrained transformer
        usable while needing only a small input adapter.
        """

        family = "bert"

        def __init__(self) -> None:
            super().__init__()
            # Temporal stride of the convolutional front-end that turns a long
            # EEG trial into a manageable sequence for the transformer.
            self.frontend_stride = int(config.extra.get("frontend_stride", 16))
            self.input_conv = nn.Conv1d(
                config.n_channels,
                config.d_model,
                kernel_size=9,
                stride=self.frontend_stride,
                padding=4,
            )
            self.input_norm = nn.GroupNorm(1, config.d_model)

            if default_name in {"config", "from_config", ""}:
                # Build the architecture from a fresh config, no download.
                # This is what makes conv-bert usable on a machine with no
                # network and no cached weights: the encoder is then trained
                # from scratch, which is an honest (if weaker) baseline than
                # refusing to run at all.
                bert_config = BertConfig(
                    vocab_size=21128,
                    hidden_size=config.d_model,
                    num_hidden_layers=int(config.extra.get("n_layers", 4)),
                    num_attention_heads=max(1, config.d_model // 32),
                    intermediate_size=config.d_model * 4,
                )
                self.bert = BertModel(bert_config)
            else:
                try:
                    bert_config = BertConfig.from_pretrained(default_name)
                    self.bert = BertModel.from_pretrained(default_name, config=bert_config)
                except OSError as exc:
                    raise BackboneError(
                        f"could not load BERT checkpoint {default_name!r}. Pass "
                        "pretrained='config' to build it without downloading, or "
                        "a local path to use local weights."
                    ) from exc

            hidden = int(self.bert.config.hidden_size)
            # BERT has a fixed maximum sequence length, and EEG trials can
            # exceed it. Instead of failing at runtime we pool the conv
            # features down to a length BERT accepts. `pool_factor` is fixed at
            # construction time so the CTC length mapping stays exact.
            self.max_frames = int(getattr(self.bert.config, "max_position_embeddings", 512))
            # Pooling is adaptive: a fixed factor cannot suit both 4 s and 60 s
            # trials, so the factor is derived per batch and the CTC length
            # mapping mirrors that same arithmetic.
            fixed_pool = config.extra.get("pool_factor")
            self.pool_factor = None if fixed_pool is None else max(1, int(fixed_pool))
            # Adapter: conv features -> BERT's embedding space.
            self.adapter = nn.Sequential(
                nn.Linear(config.d_model, hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            self._d_model = hidden
            self._hidden = hidden

            if config.freeze_backbone:
                for param in self.bert.parameters():
                    param.requires_grad = False

        @property
        def output_stride(self) -> int:
            """Base samples per conv frame, before adaptive pooling.

            The effective ratio grows with input length when pooling kicks in,
            so callers must use :meth:`input_lengths_to_output` rather than
            dividing by this value.
            """
            if self.pool_factor is not None:
                return self.frontend_stride * self.pool_factor
            return self.frontend_stride

        @property
        def d_model(self) -> int:
            return self._d_model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.ndim != 3:
                raise ValueError(f"expected (batch, channels, time), got {tuple(x.shape)}")
            if x.shape[1] != config.n_channels:
                raise ValueError(f"expected {config.n_channels} channels, got {x.shape[1]}")
            features = self.input_norm(self.input_conv(x))
            factor = self.pool_factor or self._pool_factor_for(features.shape[-1])
            if factor > 1:
                features = nn.functional.avg_pool1d(features, kernel_size=factor, stride=factor)
            embeddings = self.adapter(features.transpose(1, 2))
            out = self.bert(inputs_embeds=embeddings)
            return out.last_hidden_state

        def _pool_factor_for(self, n_frames: int) -> int:
            """Smallest pooling factor that brings ``n_frames`` within BERT's limit."""
            if n_frames <= self.max_frames:
                return 1
            return int(math.ceil(n_frames / self.max_frames))

        def input_lengths_to_output(self, lengths: torch.Tensor) -> torch.Tensor:
            """Replay the exact arithmetic: conv, then optional pooling.

            The two stages compose because both use floor division and the
            pooling has no padding.
            """
            kernel, stride, padding = 9, self.frontend_stride, 4
            frames = (
                torch.div(lengths + 2 * padding - kernel, stride, rounding_mode="floor") + 1
            ).clamp(min=1)
            if self.pool_factor is not None:
                if self.pool_factor > 1:
                    frames = torch.div(frames, self.pool_factor, rounding_mode="floor").clamp(min=1)
                return frames.to(torch.long)
            # Adaptive pooling: mirror _pool_factor_for per element.
            factors = torch.ceil(frames / self.max_frames).clamp(min=1)
            pooled = torch.div(frames, factors, rounding_mode="floor").clamp(min=1)
            return pooled.to(torch.long)

        def trainable_parameter_count(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    return BertCTCBackbone


class _ResidualConvBlock(nn.Module):
    """Depthwise-separable residual convolution with optional temporal stride."""

    def __init__(self, d_model: int, kernel_size: int, stride: int, dropout: float) -> None:
        super().__init__()
        self.stride = stride
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.pointwise = nn.Conv1d(d_model, d_model, 1)
        self.norm = nn.GroupNorm(1, d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.drop(self.act(self.norm(self.pointwise(self.depthwise(x)))))
        residual = x
        if self.stride > 1:
            residual = nn.functional.avg_pool1d(
                residual, kernel_size=self.stride, stride=self.stride
            )
        # Convolution padding can emit one extra frame; trim both branches to a
        # common length so the residual addition is always well defined.
        if residual.shape[-1] != y.shape[-1]:
            n = min(residual.shape[-1], y.shape[-1])
            residual, y = residual[..., :n], y[..., :n]
        return residual + y


class NeuroProcessor(nn.Module):
    """Convolutional EEG encoder producing continuous features.

    Mirrors NeuroLM's ``NeuralTransformer`` role: it turns raw EEG into a short
    sequence of feature vectors that a language model can consume. When used as
    a *processor* we keep it continuous (rather than VQ-quantised) so the whole
    path stays differentiable end to end for CTC.
    """

    def __init__(
        self,
        n_channels: int,
        d_model: int,
        *,
        n_blocks: int = 4,
        kernel_size: int = 9,
        input_stride: int = 8,
        block_stride: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_stride = input_stride
        self.block_stride = block_stride
        self.n_blocks = n_blocks
        self.input_conv = nn.Conv1d(
            n_channels, d_model, kernel_size, stride=input_stride, padding=kernel_size // 2
        )
        self.norm = nn.GroupNorm(1, d_model)
        self.blocks = nn.ModuleList(
            _ResidualConvBlock(
                d_model,
                kernel_size,
                stride=block_stride if i > 0 else 1,
                dropout=dropout,
            )
            for i in range(n_blocks)
        )

    @property
    def output_stride(self) -> int:
        stride = self.input_stride
        for block in self.blocks:
            stride *= block.stride
        return stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(self.input_conv(x))
        for block in self.blocks:
            y = block(y)
        return y.transpose(1, 2)

    def input_lengths_to_output(self, lengths: torch.Tensor) -> torch.Tensor:
        return resolve_length_map(lengths, self.output_stride).to(torch.long)


def make_neurolm_backbone(config: BackboneConfig) -> type[nn.Module]:
    """Build the NeuroLM-style backbone class.

    Two sub-modes share this code, selected by ``config.extra["processor"]``:

    ``"pretrained"``
        Load NeuroLM and reuse its (frozen) VQ tokenizer as the NeuroProcessor,
        training an adapter into a causal LM. This is NeuroLM's own design.
    ``"scratch"``
        Train our own :class:`NeuroProcessor` from scratch instead, keeping the
        same adapter/causal-LM structure.
    """
    try:
        from transformers import AutoConfig, AutoModel
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise BackboneError(
            "NeuroLM modes require the transformers package; install it with `uv add transformers`"
        ) from exc

    lm_name = str(config.extra.get("lm_name", "gpt2"))
    processor_kind = str(config.extra.get("processor", "scratch"))
    eeg_tokens = int(config.extra.get("eeg_tokens", 256))

    class NeuroLMBackbone(nn.Module):
        """Frozen-or-trained EEG processor plus adapter into a causal LM."""

        family = "neurolm"

        def __init__(self) -> None:
            super().__init__()
            self.processor_kind = processor_kind
            if processor_kind == "scratch":
                self.processor = NeuroProcessor(
                    config.n_channels,
                    config.d_model,
                    dropout=config.dropout,
                )
            else:
                try:
                    from bci_eeg_to_text.neurolm import load_neuro_processor
                except ImportError as exc:  # pragma: no cover
                    raise BackboneError(f"cannot load NeuroLM processor: {exc}") from exc
                self.processor = load_neuro_processor(
                    config.pretrained, config.n_channels, config.d_model
                )
                for param in self.processor.parameters():
                    param.requires_grad = False

            try:
                lm_config = AutoConfig.from_pretrained(lm_name)
                self.lm = AutoModel.from_pretrained(lm_name, config=lm_config)
            except OSError as exc:
                raise BackboneError(
                    f"could not load causal LM {lm_name!r}; pass extra['lm_name']"
                ) from exc

            hidden = int(getattr(self.lm.config, "hidden_size", config.d_model))
            self._d_model = hidden
            # The adapter is the analogue of NeuroLM's encode_transform_layer:
            # a small learned map from processor features into LM embeddings.
            self.adapter = nn.Sequential(
                nn.Linear(config.d_model, hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(hidden, hidden),
            )
            # Learned positional embedding over EEG frames, as NeuroLM does.
            self.pos_embed = nn.Embedding(eeg_tokens, hidden)

            if config.freeze_backbone:
                for param in self.lm.parameters():
                    param.requires_grad = False

        @property
        def output_stride(self) -> int:
            return self.processor.output_stride

        @property
        def d_model(self) -> int:
            return self._d_model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.ndim != 3:
                raise ValueError(f"expected (batch, channels, time), got {tuple(x.shape)}")
            features = self.processor(x)
            embeddings = self.adapter(features)
            frames = embeddings.shape[1]
            positions = (
                torch.arange(frames, device=embeddings.device) % self.pos_embed.num_embeddings
            )
            embeddings = embeddings + self.pos_embed(positions).unsqueeze(0)

            outputs = self.lm(inputs_embeds=embeddings)
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None:
                raise BackboneError(
                    f"{lm_name} did not return last_hidden_state; it cannot be used "
                    "as a feature extractor"
                )
            return hidden

        def input_lengths_to_output(self, lengths: torch.Tensor) -> torch.Tensor:
            return self.processor.input_lengths_to_output(lengths)

        def trainable_parameter_count(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    return NeuroLMBackbone


class ConvTransformerBackbone(nn.Module):
    """Convolutional front-end plus transformer encoder.

    The convolution turns a long raw trial into a short sequence; the transformer
    then attends over the whole trial at once, so each frame's prediction is
    informed by the whole sentence. CTC decodes all characters in a single
    forward pass, with no autoregressive loop.
    """

    family = "conv-transformer"

    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        from bci_eeg_to_text.conv_transformer import (
            ConvTransformerConfig,
            ConvTransformerEncoder,
        )

        extra = config.extra or {}
        self._encoder = ConvTransformerEncoder(
            ConvTransformerConfig(
                n_channels=config.n_channels,
                d_model=config.d_model,
                n_heads=int(extra.get("n_heads", 8)),
                n_layers=int(extra.get("n_layers", 4)),
                dropout=config.dropout,
                max_frames=int(extra.get("max_frames", 512)),
            )
        )

    @property
    def output_stride(self) -> int:
        return self._encoder.output_stride

    @property
    def d_model(self) -> int:
        return self._encoder.d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._encoder(x)

    def input_lengths_to_output(self, lengths: torch.Tensor) -> torch.Tensor:
        return self._encoder.input_lengths_to_output(lengths)

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_backbone(family: str, config: BackboneConfig) -> Backbone:
    """Instantiate a backbone by family name."""
    if family in {"conv", "scratch"}:
        return ConvBackbone(config)
    if family in {"conv-transformer", "transformer"}:
        return ConvTransformerBackbone(config)
    if family == "bert":
        return make_bert_backbone(config)()

    raise BackboneError(f"unknown backbone family {family!r}; expected one of {TRAINING_MODES}")


def build_ctc_model(
    mode: str,
    *,
    n_channels: int,
    n_classes: int,
    d_model: int = 256,
    pretrained: str | None = None,
    freeze_backbone: bool = False,
    dropout: float = 0.1,
    extra: dict[str, object] | None = None,
) -> nn.Module:
    """Assemble a backbone plus a CTC head, ready for training.

    ``mode`` is one of :data:`TRAINING_MODES`. The returned module accepts
    ``(B, C, T)`` EEG and returns ``(B, T', n_classes)`` logits, so the existing
    training and decoding code works unchanged for every mode.
    """
    if mode not in TRAINING_MODES:
        raise BackboneError(f"unknown mode {mode!r}; expected one of {TRAINING_MODES}")
    config = BackboneConfig(
        n_channels=n_channels,
        d_model=d_model,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
        extra=dict(extra or {}),
    )

    if mode == "scratch":
        backbone: Backbone = ConvBackbone(config)
    elif mode == "conv-transformer":
        backbone = ConvTransformerBackbone(config)
    elif mode == "bert-ctc":
        backbone = make_bert_backbone(config)()
    elif mode == "neurolm-sft":
        # Mode 1 reuses the full NeuroLM model; only the head is new.
        merged = {**config.extra, "processor": "pretrained"}
        backbone = make_neurolm_backbone(BackboneConfig(**{**config.__dict__, "extra": merged}))()
    else:  # neurolm-adapter
        merged = {**config.extra, "processor": config.extra.get("processor", "scratch")}
        backbone = make_neurolm_backbone(BackboneConfig(**{**config.__dict__, "extra": merged}))()

    return CTCModel(backbone, n_classes)


class CTCModel(nn.Module):
    """A backbone plus a linear CTC head, with length bookkeeping."""

    def __init__(self, backbone: Backbone, n_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.n_classes = n_classes
        self.head = nn.Linear(backbone.d_model, n_classes)

    @property
    def output_stride(self) -> int:
        return self.backbone.output_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C, T)`` EEG -> ``(B, T', n_classes)`` logits."""
        return self.head(self.backbone(x))

    def input_lengths_to_output(self, lengths: torch.Tensor) -> torch.Tensor:
        return self.backbone.input_lengths_to_output(lengths).to(torch.long)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def parameter_summary(self) -> dict[str, float]:
        """Total and trainable parameter counts, for logging."""
        total = sum(p.numel() for p in self.parameters())
        trainable = self.trainable_parameter_count()
        return {
            "total": float(total),
            "trainable": float(trainable),
            "frozen": float(total - trainable),
            "trainable_fraction": (trainable / total) if total else 0.0,
        }


def resolve_length_map(lengths: torch.Tensor, stride: int) -> torch.Tensor:
    """Generic ``floor((n - 1) / stride) + 1`` frame count.

    Returns an integer tensor: ``CTCLoss`` rejects non-integral lengths, and a
    float tensor from ``rounding_mode="floor"`` division would slip through
    silently until the loss call.
    """
    if stride <= 1:
        return lengths.to(torch.long)
    return torch.div(lengths - 1, stride, rounding_mode="floor") + 1


def estimate_output_frames(n_samples: int, stride: int) -> int:
    """Frames a ``stride``-downsampling backbone yields for ``n_samples``."""
    if n_samples <= 0 or stride <= 0:
        return 0
    return int(math.floor((n_samples - 1) / stride)) + 1
