#!/usr/bin/env python
"""Train the CTC EEG-to-text model on real data with accelerate.

Streams trials straight from the mounted corpora (nothing is copied), logging to
both MLflow and TensorBoard. Refuses to run on CPU unless explicitly allowed.

    python scripts/train_eeg_ctc.py --epochs 20 --out runs/baseline
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from torch.utils.data import DataLoader, Sampler  # noqa: E402

from bci_eeg_to_text.accelerated import (  # noqa: E402
    Accelerator,
    TrainArguments,
    describe_dataset,
    run_training,
)
from bci_eeg_to_text.backbones import build_ctc_model  # noqa: E402
from bci_eeg_to_text.streaming import (  # noqa: E402
    StreamingTrialDataset,
    collate,
    index_chineseeeg,
    split_by_subject,
)
from bci_eeg_to_text.tokenizer import CharTokenizer  # noqa: E402

DEFAULT_ROOT = Path("eeg-text-bci-data/bci-eeg/ds004952")


class GroupedBatchSampler(Sampler):
    """Yield batches of trial indices grouped by run, in shuffled run order."""

    def __init__(
        self, run_of: list[str], batch_size: int, *, seed: int = 0, drop_last: bool = False
    ):
        self.buckets: dict[str, list[int]] = {}
        for i, key in enumerate(run_of):
            self.buckets.setdefault(key, []).append(i)
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.drop_last = drop_last

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        import random

        rng = random.Random(self.seed + self.epoch)
        runs = list(self.buckets)
        rng.shuffle(runs)
        batch: list[int] = []
        for run in runs:
            idx = self.buckets[run][:]
            rng.shuffle(idx)
            for i in idx:
                batch.append(i)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        total = sum(len(v) for v in self.buckets.values())
        return total // self.batch_size if self.drop_last else -(-total // self.batch_size)


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--out", type=Path, default=Path("runs/ctc"))
    p.add_argument("--run-name", default="ctc-gpu")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--n-samples", type=int, default=1152)
    p.add_argument("--decimate", type=int, default=1)
    p.add_argument("--mode", default="scratch", choices=["scratch", "bert-ctc"])
    p.add_argument("--pretrained", default=None)
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--mixed-precision", default="no", choices=["no", "fp16", "bf16"])
    p.add_argument("--subjects", nargs="*", default=None)
    p.add_argument("--valid-fraction", type=float, default=0.2)
    p.add_argument("--max-minutes", type=float, default=0.0)
    p.add_argument("--max-trials", type=int, default=None, help="cap for a quick run")
    p.add_argument("--allow-cpu", action="store_true", help="permit CPU training (very slow)")
    p.add_argument("--no-mlflow", action="store_true")
    return p.parse_args()


def main() -> int:
    a = parse()
    started = time.time()

    print(f"indexing {a.data_root} ...", flush=True)
    trials = index_chineseeeg(a.data_root, subjects=a.subjects, verbose=False)
    if not trials:
        print("no trials found; is the dataset mounted?", file=sys.stderr)
        return 1
    if a.max_trials:
        trials = trials[: a.max_trials]
    train_trials, valid_trials = split_by_subject(trials, valid_fraction=a.valid_fraction)
    print(
        f"indexed {len(trials)} trials in {time.time() - started:.1f}s: "
        f"{len(train_trials)} train / {len(valid_trials)} valid "
        f"({describe_dataset(train_trials)['n_subjects']} train subjects)",
        flush=True,
    )

    tokenizer = CharTokenizer.build([t.text for t in train_trials])
    print(f"vocabulary: {len(tokenizer)} classes (train text only)", flush=True)

    train_ds = StreamingTrialDataset(
        train_trials, tokenizer, n_samples=a.n_samples, decimate=a.decimate
    )
    valid_ds = StreamingTrialDataset(
        valid_trials, tokenizer, n_samples=a.n_samples, decimate=a.decimate
    )
    run_of = [str(t.run_path) for t in train_trials]
    sampler = GroupedBatchSampler(run_of, a.batch_size, seed=0)

    acc = Accelerator(mixed_precision=a.mixed_precision, allow_cpu=a.allow_cpu)
    print(f"accelerate device: {acc.device}", flush=True)

    n_channels = int(train_ds[0]["eeg"].shape[0])
    model = build_ctc_model(
        a.mode,
        n_channels=n_channels,
        n_classes=len(tokenizer),
        d_model=a.d_model,
        pretrained=a.pretrained,
        freeze_backbone=a.freeze_backbone,
    )
    summary = model.parameter_summary()
    print(
        f"model {a.mode}: {summary['total']:,.0f} params, {summary['trainable']:,.0f} trainable",
        flush=True,
    )

    # num_workers=0: the dataset holds open file handles, which do not fork well,
    # and the sampler keeps I/O sequential so a single thread already saturates.
    train_loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate, num_workers=0)
    valid_loader = DataLoader(
        valid_ds, batch_size=a.batch_size, shuffle=False, collate_fn=collate, num_workers=0
    )

    tb_writer = None
    mlflow_mod = None
    mlflow_run = None
    if acc.is_main:
        from torch.utils.tensorboard import SummaryWriter

        tb_writer = SummaryWriter(str(a.out / "tb"))
        if not a.no_mlflow:
            import mlflow

            mlflow_mod = mlflow
            # MLflow 3 blocks the filesystem store; SQLite is the supported
            # local backend and keeps everything in one file beside the run.
            a.out.mkdir(parents=True, exist_ok=True)
            db = (a.out / "mlflow.db").resolve()
            mlflow.set_tracking_uri(f"sqlite:///{db}")
            mlflow.set_experiment("eeg-ctc")
            mlflow_run = mlflow.start_run(run_name=a.run_name)
            mlflow.log_params(
                {
                    "mode": a.mode,
                    "d_model": a.d_model,
                    "n_blocks": a.n_blocks,
                    "batch_size": a.batch_size,
                    "lr": a.lr,
                    "epochs": a.epochs,
                    "n_samples": a.n_samples,
                    "train_trials": len(train_trials),
                    "valid_trials": len(valid_trials),
                    "classes": len(tokenizer),
                    "channels": n_channels,
                    "device": str(acc.device),
                    "params_total": int(summary["total"]),
                    "dataset": describe_dataset(trials)["subjects"],
                }
            )

    def observer(name: str, value: float, step: int, kind: str) -> None:
        if tb_writer and kind in {"tb", "both"}:
            tb_writer.add_scalar(name, value, step)
        if mlflow_mod and kind in {"mlflow", "both"} and name != "train/loss_step":
            mlflow_mod.log_metric(name.replace("/", "_"), value, step=step)

    train_args = TrainArguments(
        out_dir=a.out,
        run_name=a.run_name,
        mode=a.mode,
        d_model=a.d_model,
        n_blocks=a.n_blocks,
        n_samples=a.n_samples,
        decimate=a.decimate,
        batch_size=a.batch_size,
        grad_accum=a.grad_accum,
        learning_rate=a.lr,
        epochs=a.epochs,
        mixed_precision=a.mixed_precision,
        max_minutes=a.max_minutes,
        pretrained=a.pretrained,
        freeze_backbone=a.freeze_backbone,
    )

    try:
        result = run_training(
            args=train_args,
            accelerator=acc,
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            tokenizer=tokenizer,
            trial_counts={"train": len(train_trials), "valid": len(valid_trials)},
            observer=observer,
        )
    finally:
        if tb_writer:
            tb_writer.close()
        if mlflow_run and mlflow_mod:
            if "result" in dir():
                mlflow_mod.log_metric("total_seconds", result["seconds"])
                mlflow_mod.log_artifact(str(a.out / "metrics.json"))
                if (a.out / "last.pt").exists():
                    mlflow_mod.log_artifact(str(a.out / "last.pt"))
            mlflow_mod.end_run()

    if acc.is_main and "result" in dir():
        hist = result["history"]
        print("\n=== training complete ===")
        print(f"steps {result['steps']}, {result['seconds']:.0f}s")
        if hist:
            best = min(hist, key=lambda h: h["cer"])
            print(
                f"best epoch {int(best['epoch'])}: cer={best['cer']:.3f} "
                f"acc={best['accuracy']:.3f} valid_loss={best['valid_loss']:.3f}"
            )
        print(f"artifacts: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
