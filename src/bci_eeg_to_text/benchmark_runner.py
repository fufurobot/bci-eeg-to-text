"""Execute the EEG-to-text benchmark on real, streamed data.

Training here is deliberately tiny: ``--smoke`` gives ten runs and a handful of
epochs per fold. The point is to prove the pipeline end to end -- index, stream,
train, decode, score -- and produce a number that is honest about being a smoke
test, not to produce a publishable score.

Each experiment implements :class:`Experiment`: it builds a fresh model for a
fold, trains on that fold's runs, and transcribes the test runs. A model whose
weights or code are unavailable becomes a labelled skip, never a zero.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch


class Experiment(Protocol):
    """A training-and-decoding recipe for one benchmark row."""

    name: str
    kind: str

    def prepare(self, train_rows, *, tokenizer, device, epochs, batch_size) -> None:
        """Train on ``train_rows`` for this fold."""

    def transcribe(self, rows) -> list[str]:
        """Decode one string per row, in order."""


@dataclass
class FoldResult:
    fold: int
    model: str
    cer: float
    exact_match: float
    n: int
    seconds: float


def _group_rows_by_run(rows) -> dict[str, list]:
    out: dict[str, list] = {}
    for row in rows:
        out.setdefault(str(row.run_path), []).append(row)
    return out


class ConvTransformerExperiment:
    """Our model: conv front-end + transformer, CTC, one forward pass."""

    name = "conv-transformer"
    kind = "ctc"

    def __init__(
        self, *, n_samples: int = 1152, d_model: int = 64, n_blocks: int = 2, lr: float = 3e-3
    ) -> None:
        self.n_samples = n_samples
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.lr = lr
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cpu")

    def prepare(self, train_rows, *, tokenizer, device, epochs, batch_size) -> None:
        from bci_eeg_to_text.streaming import StreamingTrialDataset, collate

        self.tokenizer = tokenizer
        self.device = device
        n_channels = 128
        head = torch.nn.Linear(self.d_model, len(tokenizer))
        model = torch.nn.Sequential()  # placeholder, replaced below
        from bci_eeg_to_text.backbones import build_ctc_model

        model = build_ctc_model(
            "conv-transformer",
            n_channels=n_channels,
            n_classes=len(tokenizer),
            d_model=self.d_model,
            extra={"n_layers": self.n_blocks, "n_heads": 4},
        ).to(device)
        del head

        dataset = StreamingTrialDataset(train_rows, tokenizer, n_samples=self.n_samples)
        optimiser = torch.optim.AdamW(model.parameters(), lr=self.lr)
        criterion = torch.nn.CTCLoss(blank=tokenizer.blank, reduction="mean")
        model.train()
        for _ in range(max(1, epochs)):
            order = list(range(len(dataset)))
            # Runs are streamed in order so each recording is opened once.
            order.sort(key=lambda i: str(train_rows[i].run_path))
            for start in range(0, len(order), batch_size):
                chunk = order[start : start + batch_size]
                batch = collate([dataset[i] for i in chunk])
                optimiser.zero_grad(set_to_none=True)
                logits = model(batch["eeg"].to(device))
                loss = criterion(
                    logits.log_softmax(-1).transpose(0, 1),
                    batch["targets"].to(device),
                    model.input_lengths_to_output(batch["input_lengths"].to(device)),
                    batch["target_lengths"].to(device),
                )
                if torch.isfinite(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimiser.step()
        model.eval()
        self.model = model

    def transcribe(self, rows) -> list[str]:
        from bci_eeg_to_text.backbones import CTCModel  # noqa: F401  (type only)
        from bci_eeg_to_text.decode import greedy_decode
        from bci_eeg_to_text.streaming import StreamingTrialDataset

        dataset = StreamingTrialDataset(
            rows,
            self.tokenizer,
            n_samples=self.n_samples,
            cache=self._reuse_cache(rows),
        )
        texts: list[str] = []
        ordered = sorted(range(len(dataset)), key=lambda i: str(rows[i].run_path))
        with torch.no_grad():
            for start in range(0, len(ordered), 8):
                chunk = ordered[start : start + 8]
                eeg = torch.stack([dataset[i]["eeg"] for i in chunk]).to(self.device)
                logits = self.model(eeg)
                lengths = self.model.input_lengths_to_output(
                    torch.tensor([eeg.shape[-1]] * len(chunk))
                )
                decoded = greedy_decode(logits, self.tokenizer, lengths)
                pairs = sorted(zip(chunk, decoded, strict=True))
                texts.extend([text for _, text in pairs])
        return texts

    def _reuse_cache(self, rows):
        from bci_eeg_to_text.streaming import RawRunCache

        return RawRunCache(capacity=4)


class ConvBertExperiment:
    """Ours: conv front-end + BERT encoder + CTC, fine-tuned in process.

    Priority row for the smoke test: BERT-base is ~110M parameters, so it fits
    this machine's 3 GB RAM / 4 GB VRAM where NeuroLM-XL (20 GB) and
    Mistral-7B (21 GB) do not. Weights come from a local folder when available
    and are otherwise initialised from a small config, so the row always runs.
    """

    name = "conv-bert"
    kind = "ctc"

    def __init__(
        self,
        *,
        n_samples: int = 1152,
        d_model: int = 128,
        n_layers: int = 2,
        lr: float = 1e-3,
        pretrained: str | None = None,
        freeze_backbone: bool = False,
    ) -> None:
        self.n_samples = n_samples
        self.d_model = d_model
        self.n_layers = n_layers
        self.lr = lr
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cpu")

    def prepare(self, train_rows, *, tokenizer, device, epochs, batch_size) -> None:
        from bci_eeg_to_text.backbones import build_ctc_model
        from bci_eeg_to_text.streaming import StreamingTrialDataset, collate

        self.tokenizer = tokenizer
        self.device = device
        model = build_ctc_model(
            "bert-ctc",
            n_channels=128,
            n_classes=len(tokenizer),
            d_model=self.d_model,
            pretrained=self.pretrained or "config",
            freeze_backbone=self.freeze_backbone,
            extra={"n_layers": self.n_layers},
        ).to(device)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimiser = torch.optim.AdamW(trainable, lr=self.lr)
        criterion = torch.nn.CTCLoss(blank=tokenizer.blank, reduction="mean")
        dataset = StreamingTrialDataset(train_rows, tokenizer, n_samples=self.n_samples)
        model.train()
        for _ in range(max(1, epochs)):
            order = sorted(range(len(dataset)), key=lambda i: str(train_rows[i].run_path))
            for start in range(0, len(order), batch_size):
                batch = collate([dataset[i] for i in order[start : start + batch_size]])
                optimiser.zero_grad(set_to_none=True)
                logits = model(batch["eeg"].to(device))
                loss = criterion(
                    logits.log_softmax(-1).transpose(0, 1),
                    batch["targets"].to(device),
                    model.input_lengths_to_output(batch["input_lengths"].to(device)),
                    batch["target_lengths"].to(device),
                )
                if torch.isfinite(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable, 5.0)
                    optimiser.step()
        model.eval()
        self.model = model

    def transcribe(self, rows) -> list[str]:
        from bci_eeg_to_text.decode import greedy_decode
        from bci_eeg_to_text.streaming import StreamingTrialDataset

        dataset = StreamingTrialDataset(rows, self.tokenizer, n_samples=self.n_samples)
        texts: list[str] = []
        ordered = sorted(range(len(dataset)), key=lambda i: str(rows[i].run_path))
        with torch.no_grad():
            for start in range(0, len(ordered), 8):
                chunk = ordered[start : start + 8]
                eeg = torch.stack([dataset[i]["eeg"] for i in chunk]).to(self.device)
                logits = self.model(eeg)
                lengths = self.model.input_lengths_to_output(
                    torch.tensor([eeg.shape[-1]] * len(chunk))
                )
                decoded = greedy_decode(logits, self.tokenizer, lengths)
                texts.extend(t for _, t in sorted(zip(chunk, decoded, strict=True)))
        return texts


class EchoExperiment:
    """ECHO: reuse the published FAST encoder, fine-tune connect + Whisper.

    The published checkpoint was pretrained for SEED/PhysioNet tasks, so its
    output vocabulary is task labels rather than Chinese text. Following the
    chosen approach, the encoder is reused and the projection and decoder are
    fine-tuned here. If the architecture cannot be instantiated from the local
    code, the row is reported as a skip with the reason.
    """

    name = "echo"
    kind = "seq2seq"

    def __init__(self, *, code_dir: Path, ckpt: Path, n_samples: int = 500) -> None:
        self.code_dir = code_dir
        self.ckpt = ckpt
        self.n_samples = n_samples
        self.error: str | None = None

    def prepare(self, train_rows, *, tokenizer, device, epochs, batch_size) -> None:
        if not (self.code_dir / "FAST").exists():
            self.error = f"ECHO code missing at {self.code_dir}"
            return
        try:
            import sys

            sys.path.insert(0, str(self.code_dir))
            import importlib

            fast_v2 = importlib.import_module("FAST.FAST_v2")
            self._tower = fast_v2.Tower  # noqa: SLF001 - published API name
        except Exception as exc:  # noqa: BLE001 - reported as a skip
            self.error = f"could not import ECHO's FAST tower: {type(exc).__name__}: {exc}"

    def transcribe(self, rows) -> list[str]:
        if self.error or not hasattr(self, "_tower"):
            raise RuntimeError(self.error or "ECHO was not prepared")
        raise RuntimeError("ECHO transcribe is not wired yet")


@dataclass
class SkipRow:
    model: str
    kind: str
    reason: str


def _load_rows(dataset: str, data_root: Path | None):
    from bci_eeg_to_text.streaming import index_chineseeeg

    if dataset != "chineseeeg":
        raise NotImplementedError(
            f"dataset {dataset!r}: only chineseeeg streams text rows; "
            "chisco support is tracked separately"
        )
    root = data_root or Path("e-text-bci-data/bci-eeg/ds004952")
    return index_chineseeeg(root, verbose=False)


def _local_bert() -> str | None:
    """A BERT folder on disk, if one exists; otherwise None (build from config)."""
    for candidate in (
        Path.home() / "hf-models" / "bert-base-chinese",
        Path.home() / "hf-models" / "bert-base-multilingual-cased",
        Path("hf-models") / "bert-base-chinese",
    ):
        if candidate.exists():
            return str(candidate)
    return None


def _dataset_path(dataset: str, data_root: Path | None) -> Path:
    if data_root is not None:
        return data_root
    return Path("eeg-text-bci-data/bci-eeg/ds004952")


def _rows_with_ids(rows):
    """Add the ``subject``/``run_id`` fields the smoke selector expects."""

    class Row(dict):
        def __getattr__(self, item):
            try:
                return self[item]
            except KeyError as exc:
                raise AttributeError(item) from exc

    out = []
    for row in rows:
        entry = Row(
            subject=row.subject,
            run=row.run,
            text=row.text,
            row_index=row.row_index,
            run_id=f"{row.subject}/{row.session}/{row.run}",
            run_path=row.run_path,
            obj=row,
        )
        out.append(entry)
    return out


def execute(
    *,
    model: str,
    datasets: list[str],
    smoke: bool,
    data_root: Path | None,
    out: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    device: str,
    echo_ckpt: Path,
    echo_code: Path,
    thought2text_dir: Path,
    hf_models: Path,
    use_mlflow: bool,
    max_temp: float = 80.0,
    thermal_interval: float = 10.0,
    printer=print,
    table=None,
) -> int:
    """Run the benchmark and print results. Returns a process exit code."""
    from bci_eeg_to_text import thermal
    from bci_eeg_to_text.benchmark import score
    from bci_eeg_to_text.benchmark_cli import collect_statuses
    from bci_eeg_to_text.smoke import fold_indices, select_smoke_runs
    from bci_eeg_to_text.tokenizer import CharTokenizer

    started = time.time()
    # Check before doing any work: an already-hot machine should not be loaded
    # further, and this machine's fan is broken.
    try:
        baseline = thermal.require_cool(max_temp)
    except thermal.ThermalError as exc:
        printer(f"refusing to run: {exc}")
        return 2
    printer(f"baseline {baseline.describe()}")

    guard = thermal.ThermalGuard(
        max_temp,
        interval=thermal_interval,
        on_sample=lambda s: printer(f"  {s.describe()}"),
    )

    if device == "auto":
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)
    printer(f"device: {device_obj}")

    statuses = collect_statuses(
        model,
        echo_ckpt=echo_ckpt,
        echo_code=echo_code,
        thought2text_dir=thought2text_dir,
        hf_models=hf_models,
    )
    requested = [s.name for s in statuses]
    printer(f"experiments: {', '.join(requested)}")

    all_rows: list[dict] = []
    for dataset in datasets:
        printer(f"\n=== dataset: {dataset} ===")
        try:
            rows = _load_rows(dataset, data_root)
        except Exception as exc:  # noqa: BLE001 - a missing mount is a skip
            for name in requested:
                all_rows.append({"model": name, "reason": f"{dataset}: {exc}"})
            continue
        printer(f"indexed {len(rows)} rows")
        if not rows:
            reason = (
                f"{dataset}: no rows indexed from {_dataset_path(dataset, data_root)}. "
                "The dataset mount is empty or the path is wrong; nothing was benchmarked."
            )
            for name in requested:
                all_rows.append({"model": name, "reason": reason})
            continue

        indexed = _rows_with_ids(rows)
        if smoke:
            try:
                selection = select_smoke_runs(indexed, seed=seed)
            except Exception as exc:  # noqa: BLE001 - a skip, not a crash
                for name in requested:
                    all_rows.append({"model": name, "reason": f"{dataset}: {exc}"})
                continue
            printer(
                f"smoke sample: subject={selection.subject} runs={len(selection.runs)} "
                f"texts={len(selection.texts)}"
            )
            chosen = [indexed[i] for i in selection.row_indices]
        else:
            chosen = indexed

        # Order by run so folds hold whole recordings, not interleaved rows.
        run_ids = sorted({str(r["run_id"]) for r in chosen})
        folds = fold_indices(len(run_ids), n_folds=5)
        printer(f"runs={len(run_ids)} folds={len(folds)}")

        for status in statuses:
            if not status.available:
                all_rows.append({"model": status.name, "reason": status.reason})
                continue
            if status.name not in ("conv-transformer", "conv-bert"):
                all_rows.append(
                    {
                        "model": status.name,
                        "reason": f"{dataset}: adapter not implemented yet "
                        f"({status.reason or 'pending'})",
                    }
                )
                continue

            fold_scores = []
            for fold_index, (train_idx, _valid_idx, test_idx) in enumerate(folds):
                # Between folds is the cheapest safe point to sample: each fold
                # is minutes long, and a breach aborts the whole benchmark.
                try:
                    guard.poll()
                except thermal.ThermalError as exc:
                    printer(f"aborting: {exc}")
                    printer(guard.summary())
                    return 3
                train_runs = {run_ids[i] for i in train_idx}
                test_runs = {run_ids[i] for i in test_idx}
                train_rows = [r["obj"] for r in chosen if str(r["run_id"]) in train_runs]
                test_rows = [r["obj"] for r in chosen if str(r["run_id"]) in test_runs]
                tokenizer = CharTokenizer.build([r.text for r in train_rows])
                experiment = (
                    ConvBertExperiment(pretrained=_local_bert())
                    if status.name == "conv-bert"
                    else ConvTransformerExperiment()
                )
                t0 = time.time()
                experiment.prepare(
                    train_rows,
                    tokenizer=tokenizer,
                    device=device_obj,
                    epochs=epochs,
                    batch_size=batch_size,
                )
                try:
                    guard.poll()
                except thermal.ThermalError as exc:
                    printer(f"aborting: {exc}")
                    printer(guard.summary())
                    return 3
                hypotheses = experiment.transcribe(test_rows)
                refs = [r.text for r in test_rows]
                result = score(refs, hypotheses)
                fold_scores.append((result, time.time() - t0))
                printer(
                    f"  fold {fold_index}: cer={result['cer']:.3f} "
                    f"exact={result['exact_match']:.3f} n={int(result['n'])} "
                    f"({time.time() - t0:.0f}s)"
                )
            total = sum(int(r["n"]) for r, _ in fold_scores) or 1
            cer = sum(r["cer"] * int(r["n"]) for r, _ in fold_scores) / total
            exact = sum(r["exact_match"] * int(r["n"]) for r, _ in fold_scores) / total
            all_rows.append(
                {
                    "model": status.name,
                    "cer": cer,
                    "exact_match": exact,
                    "n": float(total),
                    "kind": status.kind,
                }
            )

    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(all_rows, indent=2, ensure_ascii=False))
    printer("")
    if table is not None:
        printer(table(all_rows))
    printer(f"\n{guard.summary()}")
    printer(f"{time.time() - started:.0f}s elapsed; results in {out / 'results.json'}")
    return 0
