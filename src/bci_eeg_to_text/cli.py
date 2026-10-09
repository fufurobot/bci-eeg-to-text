"""Command-line entry points: prepare, train, evaluate, inspect."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_DATA_ROOT = "eeg-text-bci-data/bci-eeg/ds005170"


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help=f"Chisco dataset root (default: {DEFAULT_DATA_ROOT})",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bci-eeg-to-text",
        description="CTC-based EEG-to-text decoding on the Chisco dataset",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="extract aligned trials into a cache")
    _add_common(p_prepare)
    p_prepare.add_argument("--out", required=True, help="output cache directory")
    p_prepare.add_argument("--decimate", type=int, default=8, help="downsample factor")
    p_prepare.add_argument("--limit", type=int, default=None, help="only process N runs")
    p_prepare.add_argument("--subjects", nargs="*", default=None, help="restrict to these subjects")

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
    from bci_eeg_to_text.data import ExtractConfig, discover_runs, extract_run

    out_dir = Path(args.out)
    runs = discover_runs(args.data_root)
    if args.subjects:
        wanted = set(args.subjects)
        runs = [r for r in runs if any(f"/{s}/" in str(r) for s in wanted)]
    if args.limit:
        runs = runs[: args.limit]
    if not runs:
        print(f"no EDF runs found under {args.data_root}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, run in enumerate(runs, 1):
        # Name the cache after the run alone; run indices are globally unique,
        # so including the session would just duplicate information.
        cache = out_dir / f"{run.stem}.npz"
        if cache.exists():
            print(f"[{i}/{len(runs)}] cached {cache.name}")
            continue
        info = extract_run(run, args.data_root, cache, config=ExtractConfig(decimate=args.decimate))
        manifest.append(info)
        print(
            f"[{i}/{len(runs)}] {info['subject']} run {info['run']}: "
            f"{info['n_trials']} trials -> {cache.name}",
            flush=True,
        )

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {len(manifest)} new caches to {out_dir}")
    return 0


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
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "inspect": cmd_inspect,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
