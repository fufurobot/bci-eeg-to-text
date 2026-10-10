"""Loading NeuroLM components for the NeuroLM training modes.

NeuroLM (https://github.com/935963004/NeuroLM) encodes EEG with a frozen
vector-quantized tokenizer — the "NeuroProcessor" — and feeds the resulting
neural tokens to a GPT-2 decoder whose embedding matrix and LM head were
enlarged to hold an EEG vocabulary. ``encode_transform_layer`` is the adapter
between the two.

Three ways of using it are supported, matching the project's training modes:

1. ``neurolm-sft`` — reuse the full pretrained model and fine-tune it.
2. ``neurolm-adapter`` (``processor="pretrained"``) — keep the pretrained
   NeuroProcessor and causal LM, train only the adapter between them.
3. ``neurolm-adapter`` (``processor="scratch"``) — train our own processor from
   scratch against the same causal LM.

Nothing here is imported unless a NeuroLM mode is actually selected, so the
core pipeline runs without the extra dependency.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

#: Official checkpoints published by the authors on HuggingFace.
NEUROLM_HF_REPO = "Weibang/NeuroLM"

#: Variants, smallest first.
NEUROLM_VARIANTS = ("NeuroLM-B", "NeuroLM-L", "NeuroLM-XL")


class NeuroLMError(RuntimeError):
    """Raised when NeuroLM components cannot be loaded."""


def repo_root() -> Path | None:
    """Locate a local NeuroLM checkout, if one is present.

    Honours ``NEUROLM_ROOT``; otherwise looks in a few conventional places so a
    nearby clone is picked up automatically.
    """
    import os

    candidates = []
    env = os.environ.get("NEUROLM_ROOT")
    if env:
        candidates.append(Path(env))
    # Only location-independent candidates: baking in one machine's absolute
    # path would make behaviour depend on who runs it.
    candidates += [
        Path("NeuroLM"),
        Path("../NeuroLM"),
        Path("third_party/NeuroLM"),
    ]
    for candidate in candidates:
        if (candidate / "model" / "model_neurolm.py").exists():
            return candidate
    return None


def load_neuro_processor(
    checkpoint: str | None,
    n_channels: int,
    d_model: int,
) -> nn.Module:
    """Load a NeuroProcessor: NeuroLM's frozen VQ encoder.

    Falls back to an untrained processor of the right shape when no checkpoint
    is available, so the adapter mode can still be exercised end to end. The
    caller is told which happened, because training against a randomly
    initialised processor is not the same experiment.
    """
    from bci_eeg_to_text.backbones import NeuroProcessor

    if checkpoint is None:
        processor = NeuroProcessor(n_channels, d_model)
        processor.load_metadata = {"source": "untrained"}  # type: ignore[attr-defined]
        return processor

    path = Path(checkpoint)
    if not path.exists():
        raise NeuroLMError(
            f"NeuroProcessor checkpoint not found: {path}. Download a NeuroLM "
            f"release from https://huggingface.co/{NEUROLM_HF_REPO} or pass "
            "--processor-checkpoint with a local path."
        )

    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload

    try:
        from bci_eeg_to_text.neurolm_native import NeuralTransformerVQ
    except ImportError:
        # Without the upstream architecture we cannot map the weights; be
        # explicit rather than silently training on random features.
        raise NeuroLMError(
            "found a NeuroProcessor checkpoint but no implementation to load it "
            "into. Either place a NeuroLM checkout on NEUROLM_ROOT or use "
            "--processor scratch to train our own."
        ) from None

    # Infer the architecture from the checkpoint itself; assuming the published
    # shape would fail to load any variant with different dimensions.
    mapped = {k.removeprefix("VQ.encoder."): v for k, v in state.items()}
    config = _infer_vq_config(mapped, fallback_d_model=d_model)
    model = NeuralTransformerVQ(n_channels=n_channels, config=config)
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    model.load_metadata = {  # type: ignore[attr-defined]
        "source": str(path),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
    }
    for param in model.parameters():
        param.requires_grad = False
    return model


def _infer_vq_config(state: dict, *, fallback_d_model: int):
    """Derive tokenizer dimensions from a checkpoint's tensor shapes."""
    from bci_eeg_to_text.neurolm_native import VQConfig

    def shape(key: str) -> tuple[int, ...] | None:
        tensor = state.get(key)
        return tuple(tensor.shape) if tensor is not None else None

    codebook = shape("quantizer.embed.weight")
    codebook_size, codebook_dim = codebook if codebook else (8192, fallback_d_model)

    layers = {
        int(key.split(".")[1])
        for key in state
        if key.startswith("blocks.") and key.endswith("attn.weight")
    }
    n_layer = max(layers) + 1 if layers else 12

    attn = shape("blocks.0.attn.weight")
    n_embd = attn[0] // 3 if attn else 400

    pos = shape("pos_embed.weight")
    block_size = pos[0] if pos else 1024

    config = VQConfig(
        n_layer=n_layer,
        n_embd=n_embd,
        block_size=block_size,
        codebook_size=codebook_size,
        codebook_dim=codebook_dim,
    )
    patch = shape("patch_embed.proj.weight")
    if patch:
        config.patch_size = int(patch[-1])
    return config


def describe_neuro_processor(processor: nn.Module) -> str:
    """Human-readable provenance of a loaded processor."""
    meta = getattr(processor, "load_metadata", {})
    source = meta.get("source", "unknown")
    n_params = sum(p.numel() for p in processor.parameters())
    frozen = all(not p.requires_grad for p in processor.parameters())
    state = "frozen" if frozen else "trainable"
    return f"NeuroProcessor[{source}, {n_params:,} params, {state}]"
