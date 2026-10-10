"""Command-line entry points: prepare, train, evaluate, inspect."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: Parent directory holding both datasets as subdirectories.
DEFAULT_DATA_ROOT = "eeg-text-bci-data/bci-eeg"


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help=f"dataset parent directory (default: {DEFAULT_DATA_ROOT})",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bci-eeg-to-text",
        description="CTC-based EEG-to-text decoding over EEG-text datasets",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="extract aligned trials into a cache")
    _add_common(p_prepare)
    p_prepare.add_argument(
        "--dataset",
        default="chisco",
        choices=["chisco", "chineseeeg", "merged"],
        help="which corpus to read (merged reads both)",
    )
    p_prepare.add_argument("--out", required=True, help="output cache directory")
    p_prepare.add_argument("--decimate", type=int, default=8, help="downsample factor")
    p_prepare.add_argument("--limit", type=int, default=None, help="only process N runs")
    p_prepare.add_argument("--subjects", nargs="*", default=None, help="restrict to these subjects")

    p_split = sub.add_parser("split", help="show train/valid/test subject splits")
    _add_common(p_split)
    p_split.add_argument(
        "--dataset",
        default="merged",
        choices=["chisco", "chineseeeg", "merged"],
        help="which corpus(s) to split",
    )
    p_split.add_argument(
        "--level",
        default="subject",
        choices=["subject", "run"],
        help="split by participant (default) or by run (within-participant)",
    )
    p_split.add_argument("--seed", type=int, default=0)
    p_split.add_argument(
        "--ratios",
        default=None,
        help="override ratios, e.g. 'train=1,test=1' (default 3:1:1, or 1:1 for chisco)",
    )
    p_split.add_argument("--out", default=None, help="write the split as JSON here")

    p_inspect = sub.add_parser("inspect", help="report an EDF run without extracting")
    _add_common(p_inspect)
    p_inspect.add_argument("edf", help="path to an EDF file")

    p_train = sub.add_parser("train", help="train the CTC model")
    p_train.add_argument("--cache", required=True, help="cache directory from prepare")
    p_train.add_argument("--out", required=True, help="checkpoint output directory")
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--batch-size", type=int, default=8)
    p_train.add_argument("--lr", type=float, default=3e-4)
    p_train.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    p_train.add_argument("--n-samples", type=int, default=8000)
    p_train.add_argument("--d-model", type=int, default=256)
    p_train.add_argument("--n-blocks", type=int, default=4)
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument(
        "--max-steps", type=int, default=None, help="stop after N optimizer steps (smoke test)"
    )

    p_eval = sub.add_parser("evaluate", help="evaluate a checkpoint")
    p_eval.add_argument("--cache", required=True)
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--device", default=None)
    p_eval.add_argument("--n-samples", type=int, default=8000)
    p_eval.add_argument("--beam-width", type=int, default=1, help="1 means greedy")
    return parser


def _default_ratios(dataset: str) -> dict[str, int]:
    """Agreed split ratios per dataset.

    ds004952 has 10 subjects, enough for a held-out validation set; ds005170 has
    only 4 usable subjects, so it is split 1:1 train/test.
    """
    if dataset == "chisco":
        return {"train": 1, "test": 1}
    return {"train": 3, "valid": 1, "test": 1}


def _parse_ratios(spec: str | None, dataset: str) -> dict[str, int]:
    if not spec:
        return _default_ratios(dataset)
    ratios: dict[str, int] = {}
    for chunk in spec.split(","):
        if "=" not in chunk:
            raise SystemExit(f"bad --ratios entry {chunk!r}; expected name=int")
        key, _, value = chunk.partition("=")
        ratios[key.strip()] = int(value)
    return ratios


def cmd_split(args: argparse.Namespace) -> int:
    """Show (and optionally persist) a gender-balanced subject-level split."""
    from bci_eeg_to_text.sources import get_source, subjects_for_split
    from bci_eeg_to_text.splits import make_splits, validate_subject_disjoint

    names = _resolve_dataset_names(args.dataset)
    sources = [
        get_source(name, data_root=Path(args.data_root) / _root_suffix(name)) for name in names
    ]
    participants = subjects_for_split(sources)
    if not participants:
        print("no participants found; is the dataset mounted?", file=sys.stderr)
        return 1

    ratios = _parse_ratios(args.ratios, args.dataset)
    runs_per_subject = None
    if args.level == "run":
        runs_per_subject = {}
        for source in sources:
            for run in source.runs():
                key = f"{source.name}:{run.subject_id}"
                runs_per_subject.setdefault(key, []).append(run.run)

    split = make_splits(
        participants,
        ratios=ratios,
        level=args.level,
        seed=args.seed,
        runs_per_subject=runs_per_subject,
    )
    validate_subject_disjoint(split, participants)

    by_id = {p.subject_id: p for p in participants}
    print(f"dataset: {args.dataset}  level: {args.level}  ratios: {ratios}  seed: {args.seed}")
    for part, entries in split.parts.items():
        # At run level the entries are "subject/run"; keep subject and run
        # distinct so the shared participant always comes from the entry.
        subject_of = {e: (e.split("/")[0] if "/" in e else e) for e in entries}
        for subject in subject_of.values():
            if subject not in by_id:
                raise SystemExit(f"split referenced unknown subject {subject!r}")
        sexes = [by_id[subject_of[e]].sex or "?" for e in entries]
        unit = "runs" if not split.subject_level else "subjects"
        summary = f"{sexes.count('F')}F/{sexes.count('M')}M"
        print(f"\n{part} ({len(entries)} {unit}, {summary}):")
        for entry in entries:
            subject = subject_of[entry]
            label = entry if not split.subject_level else subject
            print(f"  {label} ({by_id[subject].sex or 'sex unknown'})")

    # A run-level split is legitimate but is not a generalisation estimate, so
    # say so loudly rather than letting the number be misread.
    for warning in split.warnings:
        print(f"\nWARNING: {warning}", file=sys.stderr)

    if args.out:
        payload = {
            "dataset": args.dataset,
            "level": args.level,
            "seed": args.seed,
            "ratios": ratios,
            "subject_level": split.subject_level,
            "warnings": list(split.warnings),
            "parts": {k: list(v) for k, v in split.parts.items()},
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nwrote {args.out}")
    return 0


def cmd_inspect(args: argparse.ArgumentParser) -> int:
    from bci_eeg_to_text.eeg import find_pulses, read_metadata, read_signals
    from bci_eeg_to_text.textdata import read_split, split_file_for_run

    path = Path(args.edf)
    meta = read_metadata(path)
    signals, rate = read_signals(path, ["Trigger"])
    pulses = find_pulses(signals[0])
    split = split_file_for_run(Path(args.data_root) / "textdataset", meta.run_id.run)
    texts = read_split(split)

    report = {
        "run": meta.run_id.key,
        "sample_rate_hz": rate,
        "duration_s": round(meta.duration_s, 1),
        "n_eeg_channels": meta.n_eeg_channels,
        "reward_pulses": int(len(pulses)),
        "stimulus_sentences": len(texts),
        "aligned": len(pulses) == len(texts),
        "stimulus_file": split.name,
    }
    if pulses.size:
        gaps = (pulses[1:] - pulses[:-1]) / rate if pulses.size > 1 else []
        report["mean_trial_gap_s"] = round(float(sum(gaps) / len(gaps)), 2) if len(gaps) else None
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["aligned"] else 1


def cmd_prepare(args: argparse.Namespace) -> int:
    """Extract (EEG, text) pairs from one or more dataset sources."""
    from bci_eeg_to_text.sources import get_source

    names = _resolve_dataset_names(args.dataset)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    skipped: list[tuple[str, str]] = []
    for name in names:
        source = get_source(name, data_root=Path(args.data_root) / _root_suffix(name))
        print(f"# {source.describe()}", flush=True)
        runs = source.runs()
        if args.subjects:
            wanted = set(args.subjects)
            runs = [r for r in runs if r.subject_id in wanted]
        if args.limit:
            runs = runs[: args.limit]
        if not runs:
            print(f"  no usable runs for {name}", file=sys.stderr)
            continue

        for i, run in enumerate(runs, 1):
            label = f"{name}_{run.subject_id}_{run.session or 'na'}_{run.run}"
            cache = out_dir / f"{label}.npz"
            if cache.exists():
                print(f"[{i}/{len(runs)}] cached {cache.name}", flush=True)
                continue
            try:
                report = source.extract(run, cache, decimate=args.decimate)
            except Exception as exc:  # noqa: BLE001 - report and continue the sweep
                skipped.append((f"{name}:{run.subject_id}/{run.run}", str(exc)))
                print(f"[{i}/{len(runs)}] SKIP {run.subject_id} {run.run}: {exc}", flush=True)
                continue
            if not report.ok:
                reason = report.skipped_reason or "unknown"
                skipped.append((f"{name}:{run.subject_id}/{run.run}", reason))
                print(f"[{i}/{len(runs)}] SKIP {run.subject_id} {run.run}: {reason}", flush=True)
                continue
            manifest.append(
                {
                    "dataset": name,
                    "subject": run.subject_id,
                    "run": run.run,
                    "session": run.session,
                    "n_segments": report.n_segments,
                    **report.details,
                }
            )
            print(
                f"[{i}/{len(runs)}] {run.subject_id} {run.run}: "
                f"{report.n_segments} segments -> {cache.name}",
                flush=True,
            )

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {len(manifest)} caches to {out_dir}")
    if skipped:
        print(f"skipped {len(skipped)} runs; details in skipped.json", file=sys.stderr)
        (out_dir / "skipped.json").write_text(
            json.dumps(skipped, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return 0


def _resolve_dataset_names(value: str) -> list[str]:
    """Turn ``--dataset`` into concrete source names."""
    if value == "merged":
        return ["chisco", "chineseeeg"]
    return [value]


def _root_suffix(name: str) -> str:
    """Subdirectory of the shared data root holding a given dataset."""
    return {"chisco": "ds005170", "chineseeeg": "ds004952"}.get(name, name)


def _load_dataset(cache_dir: Path, n_samples: int):
    """Load a trial dataset and tokenizer, inferring the channel count.

    The channel count is read from the cache rather than hard-coded to 128 so a
    decimated or channel-subset cache trains correctly instead of failing at the
    first forward pass.
    """
    import numpy as np

    from bci_eeg_to_text.data import TrialDataset
    from bci_eeg_to_text.tokenizer import CharTokenizer

    caches = sorted(cache_dir.glob("*.npz"))
    if not caches:
        raise SystemExit(f"no cache files in {cache_dir}; run `prepare` first")

    texts: list[str] = []
    n_channels: int | None = None
    for path in caches:
        with np.load(path, allow_pickle=True) as data:
            texts.extend(str(t) for t in data["texts"])
            if n_channels is None:
                n_channels = int(data["n_channels"])
    tokenizer = CharTokenizer.build(texts)
    if n_channels is None:
        raise SystemExit(f"cache files in {cache_dir} contain no trials")
    return TrialDataset(caches, tokenizer, n_samples=n_samples), tokenizer, n_channels


def cmd_train(args: argparse.Namespace) -> int:
    import torch

    from bci_eeg_to_text.model import EEGToTextCTC, ModelConfig
    from bci_eeg_to_text.train import (
        TrainConfig,
        evaluate,
        make_batches,
        resolve_device,
        train_one_epoch,
    )

    device = resolve_device(args.device)
    dataset, tokenizer, n_channels = _load_dataset(Path(args.cache), args.n_samples)
    print(f"device={device} trials={len(dataset)} channels={n_channels} classes={len(tokenizer)}")

    torch.manual_seed(args.seed)
    model = EEGToTextCTC(
        ModelConfig(
            n_channels=n_channels,
            n_classes=len(tokenizer),
            d_model=args.d_model,
            n_blocks=args.n_blocks,
        )
    ).to(device)

    trials = [dataset[i] for i in range(len(dataset))]
    split = max(1, int(len(trials) * 0.9))
    train_batches = make_batches(trials[:split], args.batch_size, shuffle=True, seed=args.seed)
    valid_batches = make_batches(trials[split:], args.batch_size)

    config = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(out_dir / "vocab.json")

    steps = 0
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_batches, optimizer, tokenizer, device, config)
        steps += len(train_batches)
        metrics = evaluate(model, valid_batches, tokenizer, device, max_examples=3)
        print(
            f"epoch {epoch}: train_loss={loss:.3f} valid_loss={metrics.loss:.3f} "
            f"cer={metrics.cer:.3f} acc={metrics.sentence_accuracy:.3f}"
        )
        for truth, hyp in metrics.examples:
            print(f"    ref={truth!r} hyp={hyp!r}")

        torch.save(
            {
                "model": model.state_dict(),
                "config": model.config.__dict__,
                "vocab": tokenizer.itos,
                "epoch": epoch,
            },
            out_dir / "last.pt",
        )
        if args.max_steps is not None and steps >= args.max_steps:
            print(f"reached --max-steps={args.max_steps}, stopping")
            break

    print(f"checkpoints written to {out_dir}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    import torch

    from bci_eeg_to_text.decode import (
        beam_search_decode,
        character_error_rate,
        sentence_accuracy,
    )
    from bci_eeg_to_text.model import EEGToTextCTC, ModelConfig
    from bci_eeg_to_text.train import evaluate, make_batches, move_batch_to_device, resolve_device

    device = resolve_device(args.device)
    dataset, tokenizer, _ = _load_dataset(Path(args.cache), args.n_samples)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = EEGToTextCTC(ModelConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"])

    batches = make_batches([dataset[i] for i in range(len(dataset))], 8)
    if args.beam_width <= 1:
        metrics = evaluate(model, batches, tokenizer, device, max_examples=10)
    else:
        # Greedy evaluate() gives the loss; beam search gives the reported text.
        metrics = evaluate(model, batches, tokenizer, device, max_examples=0)
        refs: list[str] = []
        hyps: list[str] = []
        model.eval()
        with torch.no_grad():
            for batch in batches:
                moved = move_batch_to_device(batch, device)
                logits = model(moved["eeg"])
                lengths = model.input_lengths_to_output(moved["input_lengths"])
                for i, text in enumerate(batch["texts"]):
                    refs.append(str(text))
                    hyps.append(
                        beam_search_decode(
                            logits[i],
                            tokenizer,
                            beam_width=args.beam_width,
                            input_length=int(lengths[i]),
                        )[0]
                    )
        metrics.cer = character_error_rate(refs, hyps)
        metrics.sentence_accuracy = sentence_accuracy(refs, hyps)
        metrics.examples = list(zip(refs, hyps, strict=True))[:10]
        metrics.n_examples = len(refs)

    print(
        json.dumps(
            {
                "loss": metrics.loss,
                "cer": metrics.cer,
                "sentence_accuracy": metrics.sentence_accuracy,
                "n_examples": metrics.n_examples,
            },
            indent=2,
        )
    )
    for truth, hyp in metrics.examples:
        print(f"ref={truth!r}\nhyp={hyp!r}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "prepare": cmd_prepare,
        "split": cmd_split,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "inspect": cmd_inspect,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
