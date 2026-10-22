"""Backend-agnostic CTC training with HuggingFace ``accelerate``.

The model and training loop are written once and run on whatever Accelerator
selects (CUDA, MPS, multi-GPU), so the same code path that trains here would
train on a larger machine. Gradients, mixed precision, gradient accumulation and
learning-rate scheduling all come from Accelerator rather than being reimplemented.

Two deliberate constraints:

* **GPU only.** ``--device cpu`` is rejected unless ``--allow-cpu`` is passed.
  A CPU run takes ~2 s per step against 16 ms on the GPU here, so silently
  falling back would turn a ten-minute job into hours and produce a result that
  looks like a modelling failure.
* **No derived data on disk.** Trials are read as random-access slices straight
  from the mounted corpora via :mod:`bci_eeg_to_text.streaming`; the run writes
  only metrics and checkpoints.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from bci_eeg_to_text.streaming import EchoTrial


@dataclass
class TrainArguments:
    """Everything that defines a run, recorded alongside the metrics."""

    out_dir: Path = Path("runs")
    run_name: str = "ctc"
    mode: str = "scratch"
    d_model: int = 128
    n_blocks: int = 4
    n_samples: int = 1152
    decimate: int = 1
    batch_size: int = 16
    grad_accum: int = 1
    learning_rate: float = 2e-3
    weight_decay: float = 1e-2
    epochs: int = 20
    warmup_ratio: float = 0.06
    grad_clip: float = 5.0
    mixed_precision: str = "no"
    log_every: int = 20
    eval_every_epochs: int = 1
    seed: int = 0
    max_minutes: float = 0.0
    pretrained: str | None = None
    freeze_backbone: bool = False
    extra: dict[str, object] = field(default_factory=dict)


class Accelerator:
    """Thin wrapper so the call sites read the same for one or many devices.

    Kept separate from the training loop so the loop can be exercised with a
    tiny fake accelerator in tests, without downloading models or needing a GPU.
    """

    def __init__(self, *, mixed_precision: str = "no", allow_cpu: bool = False) -> None:
        from accelerate import Accelerator as _Accelerator
        from accelerate.utils import set_seed

        self._acc = _Accelerator(
            mixed_precision=mixed_precision if mixed_precision != "no" else "no",
            cpu=allow_cpu,
            gradient_accumulation_steps=1,
        )
        if self._acc.device.type == "cpu" and not allow_cpu:
            raise RuntimeError(
                "accelerate selected CPU. Training on CPU here is ~125x slower "
                "(2 s/step vs 16 ms), so it is refused. Pass --allow-cpu only if "
                "that cost is intended."
            )
        set_seed(0)

    @property
    def device(self) -> torch.device:
        return self._acc.device

    @property
    def process_index(self) -> int:
        return self._acc.process_index

    @property
    def is_main(self) -> bool:
        return self._acc.is_main_process

    def prepare(self, *objects):
        return self._acc.prepare(*objects)

    def accumulate(self, model):
        return self._acc.accumulate(model)

    def backward(self, loss) -> None:
        self._acc.backward(loss)

    def wait(self) -> None:
        self._acc.wait_for_everyone()

    def save(self, state: dict, path: Path) -> None:
        self._acc.save(state, str(path))

    def unwrap(self, model):
        return self._acc.unwrap_model(model)


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float):
    """Linear warmup then cosine decay over the whole run."""
    warmup = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def ctc_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    input_lengths: torch.Tensor,
    blank: int,
) -> torch.Tensor:
    """CTC loss from ``(B, T, C)`` logits.

    ``zero_infinity`` stays off: an infinite loss means the input is shorter
    than its target, which is a bug worth surfacing rather than masking.
    """
    log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
    return F.ctc_loss(
        log_probs,
        targets,
        input_lengths.to(torch.long),
        target_lengths.to(torch.long),
        blank=blank,
        reduction="mean",
    )


@torch.no_grad()
def evaluate_ctc(
    model,
    loader,
    tokenizer,
    *,
    device: torch.device,
    blank: int,
    limit_batches: int | None = None,
) -> dict[str, float]:
    """Greedy-decode a validation loader and report loss, CER and accuracy."""
    from bci_eeg_to_text.decode import character_error_rate, greedy_decode, sentence_accuracy

    model.eval()
    total, nb = 0.0, 0
    refs: list[str] = []
    hyps: list[str] = []
    for batch in loader:
        eeg = batch["eeg"].to(device)
        logits = model(eeg)
        lengths = model.input_lengths_to_output(batch["input_lengths"].to(device))
        loss = ctc_loss(
            logits,
            batch["targets"].to(device),
            batch["target_lengths"].to(device),
            lengths,
            blank,
        )
        total += float(loss)
        nb += 1
        refs.extend(batch["texts"])
        hyps.extend(greedy_decode(logits, tokenizer, lengths))
        if limit_batches and nb >= limit_batches:
            break
    model.train()
    return {
        "valid_loss": total / max(1, nb),
        "cer": character_error_rate(refs, hyps) if refs else float("nan"),
        "accuracy": sentence_accuracy(refs, hyps) if refs else 0.0,
        "n_examples": float(nb),
        "_examples": list(zip(refs, hyps, strict=True))[:8],
    }


def run_training(
    *,
    args: TrainArguments,
    accelerator: Accelerator,
    model,
    train_loader,
    valid_loader,
    tokenizer,
    trial_counts: dict[str, int],
    observer=None,
) -> dict[str, object]:
    """Train to completion, reporting to MLflow and TensorBoard.

    ``observer`` receives ``(name, value, step, kind)`` so logging is optional
    and the loop is testable without either service running.
    """
    total_steps = len(train_loader) * args.epochs
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(optimizer, max(1, total_steps), args.warmup_ratio)

    model, optimizer, train_loader, valid_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, valid_loader, scheduler
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    step = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, nb = 0.0, 0
        for batch in train_loader:
            with accelerator.accumulate(model):
                logits = model(batch["eeg"])
                lengths = model.input_lengths_to_output(batch["input_lengths"])
                loss = ctc_loss(
                    logits,
                    batch["targets"],
                    batch["target_lengths"],
                    lengths,
                    tokenizer.blank,
                )
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"non-finite CTC loss at step {step}: the input is shorter "
                        "than the target after downsampling"
                    )
                accelerator.backward(loss)
                if args.grad_clip > 0:
                    accelerator._acc.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            running += float(loss)
            nb += 1
            step += 1
            lr = scheduler.get_last_lr()[0]
            if observer and step % args.log_every == 0:
                observer("train/loss_step", float(loss), step, "tb")
                observer("train/lr", lr, step, "both")

        if nb == 0:
            break
        metrics = {
            "train_loss": running / nb,
            "lr": scheduler.get_last_lr()[0],
            "epoch": float(epoch),
        }
        accelerator.wait()
        if accelerator.is_main and epoch % args.eval_every_epochs == 0:
            ev = evaluate_ctc(
                accelerator.unwrap(model),
                valid_loader,
                tokenizer,
                device=accelerator.device,
                blank=tokenizer.blank,
            )
            examples = ev.pop("_examples")
            metrics.update({k: float(v) for k, v in ev.items()})
            history.append(dict(metrics))
            if observer:
                for key, value in metrics.items():
                    observer(key, float(value), epoch, "both")
            elapsed = time.time() - started
            print(
                f"epoch {epoch:>3}/{args.epochs}  train={metrics['train_loss']:.3f}  "
                f"valid={metrics['valid_loss']:.3f}  cer={metrics['cer']:.3f}  "
                f"acc={metrics['accuracy']:.3f}  lr={metrics['lr']:.1e}  "
                f"{elapsed:.0f}s",
                flush=True,
            )
            for truth, hyp in examples[:3]:
                print(f"    ref={truth!r}\n    hyp={hyp!r}", flush=True)

            state = {
                "model": accelerator.unwrap(model).state_dict(),
                "train_args": {
                    k: str(v) if isinstance(v, Path) else v for k, v in asdict(args).items()
                },
                "vocab": tokenizer.itos,
                "epoch": epoch,
                "metrics": metrics,
            }
            accelerator.save(state, out / "last.pt")
            (out / "metrics.json").write_text(
                json.dumps({"history": history, "trial_counts": trial_counts}, indent=2)
            )

        if args.max_minutes and (time.time() - started) / 60 > args.max_minutes:
            print(f"stopping: {args.max_minutes} minute budget reached", flush=True)
            break

    return {"history": history, "steps": step, "seconds": time.time() - started}


def describe_dataset(trials: list[EchoTrial]) -> dict[str, object]:
    """Summary of what a run will train on, for logging and provenance."""
    subjects = sorted({t.subject for t in trials})
    runs = sorted({(t.subject, t.session, t.run) for t in trials})
    return {
        "n_trials": len(trials),
        "n_subjects": len(subjects),
        "n_runs": len(runs),
        "subjects": subjects,
        "mean_text_length": round(sum(len(t.text) for t in trials) / max(1, len(trials)), 2),
    }
