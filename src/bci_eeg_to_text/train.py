"""CTC training and evaluation loops."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from bci_eeg_to_text.data import Trial, collate
from bci_eeg_to_text.decode import character_error_rate, greedy_decode, sentence_accuracy
from bci_eeg_to_text.model import EEGToTextCTC
from bci_eeg_to_text.tokenizer import CharTokenizer


def resolve_device(preference: str | None = None) -> torch.device:
    """Pick a compute device, preferring CUDA when it is actually usable.

    ``torch.cuda.is_available()`` is the only reliable check; a CUDA build of
    PyTorch can still fail to reach the driver, and pretending otherwise turns
    a clear error into a confusing one later.
    """
    if preference is not None:
        device = torch.device(preference)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available; check the NVIDIA driver "
                "and /dev/nvidia* device nodes"
            )
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class TrainConfig:
    """Optimization settings."""

    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    seed: int = 0
    #: Teacher-forcing style schedule is unnecessary for CTC; this keeps the
    #: option to freeze the encoder for a few epochs of head warm-up.
    warmup_epochs: int = 0


@dataclass
class EpochMetrics:
    """Aggregate metrics for one pass over a dataset."""

    loss: float
    cer: float
    sentence_accuracy: float
    n_examples: int
    examples: list[tuple[str, str]] = field(default_factory=list)


def move_batch_to_device(
    batch: dict[str, np.ndarray], device: torch.device
) -> dict[str, torch.Tensor]:
    """Convert a numpy batch into device tensors, leaving texts as-is."""
    out: dict[str, torch.Tensor] = {}
    for key in ("eeg", "targets", "target_lengths", "input_lengths"):
        out[key] = torch.as_tensor(batch[key], device=device)
    return out


def compute_loss(
    model: EEGToTextCTC,
    batch: dict[str, torch.Tensor],
    criterion: nn.CTCLoss,
) -> torch.Tensor:
    """CTC loss for one batch.

    ``zero_infinity`` is left off deliberately: an infinite loss means the
    input is shorter than the target, which is a bug worth surfacing rather
    than silently zeroing.
    """
    logits = model(batch["eeg"])
    log_probs = logits.log_softmax(dim=-1).transpose(0, 1)  # CTC wants (T, N, C)
    input_lengths = model.input_lengths_to_output(batch["input_lengths"])
    return criterion(
        log_probs,
        batch["targets"],
        input_lengths,
        batch["target_lengths"],
    )


def evaluate(
    model: EEGToTextCTC,
    batches: Iterable[dict[str, np.ndarray]],
    tokenizer: CharTokenizer,
    device: torch.device,
    *,
    max_examples: int = 8,
) -> EpochMetrics:
    """Run greedy decoding over ``batches`` and report loss plus text metrics."""
    model.eval()
    criterion = nn.CTCLoss(blank=tokenizer.blank, reduction="mean")
    total_loss, n_batches, n_examples = 0.0, 0, 0
    refs: list[str] = []
    hyps: list[str] = []
    examples: list[tuple[str, str]] = []

    with torch.no_grad():
        for batch in batches:
            moved = move_batch_to_device(batch, device)
            loss = compute_loss(model, moved, criterion)
            total_loss += float(loss)
            n_batches += 1

            logits = model(moved["eeg"])
            lengths = model.input_lengths_to_output(moved["input_lengths"])
            decoded = greedy_decode(logits, tokenizer, lengths)
            truths = [str(t) for t in batch["texts"]]
            refs.extend(truths)
            hyps.extend(decoded)
            n_examples += len(truths)
            if len(examples) < max_examples:
                # Take only as many as still fit: appending a whole batch would
                # overshoot the cap whenever the batch is larger than the gap.
                room = max_examples - len(examples)
                examples.extend(list(zip(truths, decoded, strict=True))[:room])

    return EpochMetrics(
        loss=total_loss / max(1, n_batches),
        cer=character_error_rate(refs, hyps) if refs else float("nan"),
        sentence_accuracy=sentence_accuracy(refs, hyps) if refs else 0.0,
        n_examples=n_examples,
        examples=examples,
    )


def train_one_epoch(
    model: EEGToTextCTC,
    batches: Sequence[dict[str, np.ndarray]],
    optimizer: torch.optim.Optimizer,
    tokenizer: CharTokenizer,
    device: torch.device,
    config: TrainConfig,
) -> float:
    """One optimization pass; returns the mean training loss."""
    model.train()
    criterion = nn.CTCLoss(blank=tokenizer.blank, reduction="mean")
    total, n_batches = 0.0, 0

    for batch in batches:
        moved = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(model, moved, criterion)
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite CTC loss ({loss.item()}); this usually means the "
                "input is shorter than the target after downsampling"
            )
        loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        total += float(loss)
        n_batches += 1

    return total / max(1, n_batches)


def make_batches(
    trials: Sequence[Trial],
    batch_size: int,
    *,
    shuffle: bool = False,
    seed: int = 0,
) -> list[dict[str, np.ndarray]]:
    """Group trials into collated batches."""
    order = np.arange(len(trials))
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    batches = []
    for start in range(0, len(order), batch_size):
        chunk = [trials[i] for i in order[start : start + batch_size]]
        if chunk:
            batches.append(collate(chunk))
    return batches
