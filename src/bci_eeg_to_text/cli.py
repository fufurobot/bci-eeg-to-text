"""Command-line entry points: prepare, train, evaluate, inspect."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bci_eeg_to_text.backbones import TRAINING_MODES
from bci_eeg_to_text.dataconfig import (
    DATA_ROOT_ENV,
    DATA_SUBDIR_ENV,
    DATA_WRITABLE_ENV,
    DEFAULT_DATA_ROOT,
    DEFAULT_DATA_SUBDIR,
    DataConfig,
)
from bci_eeg_to_text.leakage import (
    DEFAULT_SUBJECT_MODE,
    DOCUMENT_DATASETS,
    LEAKAGE_MODES,
    SUBJECT_MODES,
    LeakageError,
    describe_subject_mode,
    describe_text_mode,
)
from bci_eeg_to_text.textstructure import DEFAULT_LEAKAGE_MODE


def _add_common(parser: argparse.ArgumentParser) -> None:
    """Shared dataset-location options.

    Defaults come from the environment (see :mod:`bci_eeg_to_text.dataconfig`)
    so nothing about the dataset layout is hard-coded, and an explicit flag
    always overrides the environment.
    """
    parser.add_argument(
        "--data-root",
        default=None,
        help=f"directory containing the corpora (env {DATA_ROOT_ENV}, default {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--data-subdir",
        default=None,
        help=f"corpus subdirectory (env {DATA_SUBDIR_ENV}, default {DEFAULT_DATA_SUBDIR})",
    )
    parser.add_argument(
        "--data-writable",
        action="store_true",
        default=None,
        help=(
            "assume the dataset root is writable and create splits inside it; "
            f"default is to write to the current directory (env {DATA_WRITABLE_ENV})"
        ),
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
    p_split.add_argument(
        "--link",
        action="store_true",
        help=(
            "materialise the split as a browsable tree of relative symlinks "
            "(no data is copied); use --data-writable to place it inside the "
            "dataset root instead of the current directory"
        ),
    )
    p_split.add_argument(
        "--no-link", dest="link", action="store_false", help="only print the split"
    )
    p_split.set_defaults(link=True)
    p_split.add_argument(
        "--subject-mode",
        default=DEFAULT_SUBJECT_MODE,
        choices=list(SUBJECT_MODES),
        help=(
            "subject leakage policy: "
            + "; ".join(f"{m} = {describe_subject_mode(m)}" for m in SUBJECT_MODES)
        ),
    )
    p_split.add_argument(
        "--leakage",
        default=DEFAULT_LEAKAGE_MODE,
        choices=list(LEAKAGE_MODES),
        help=(
            "text leakage policy: "
            + "; ".join(f"{m} = {describe_text_mode(m)}" for m in LEAKAGE_MODES)
        ),
    )
    p_split.add_argument(
        "--no-text-copy",
        dest="copy_text",
        action="store_false",
        help=(
            "do not copy small text sidecars into the split. By default they are "
            "copied next to the EEG links so the EEG-to-text pairing is "
            "inspectable with ls; recordings are always symlinked, never copied"
        ),
    )
    p_split.set_defaults(copy_text=True)
    p_split.add_argument("--out", default=None, help="also write the split as JSON here")

    p_inspect = sub.add_parser("inspect", help="report an EDF run without extracting")
    _add_common(p_inspect)
    p_inspect.add_argument("edf", help="path to an EDF file")

    p_train = sub.add_parser("train", help="train the CTC model")
    p_train.add_argument(
        "--split",
        default=None,
        help=(
            "path to a materialised split tree (from `split --link`). "
            "Preferred: the split defines train/valid/test, so subject-level "
            "and run-level splits both work and nothing leaks across parts."
        ),
    )
    p_train.add_argument(
        "--cache",
        default=None,
        help=(
            "directory of pre-extracted .npz trials. Only for quick experiments: "
            "it carries no split information, so validation falls back to a "
            "holdout from the training data. Use --split for real evaluation."
        ),
    )
    p_train.add_argument(
        "--cache-dir",
        default=None,
        help="where --split writes its extracted trials (default: <out>/cache)",
    )
    p_train.add_argument(
        "--decimate",
        type=int,
        default=8,
        help="downsample factor applied when extracting from a --split",
    )
    p_train.add_argument("--out", required=True, help="checkpoint output directory")
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--batch-size", type=int, default=8)
    p_train.add_argument("--lr", type=float, default=3e-4)
    p_train.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    p_train.add_argument("--n-samples", type=int, default=8000)
    p_train.add_argument("--d-model", type=int, default=256)
    p_train.add_argument("--n-blocks", type=int, default=4)
    p_train.add_argument(
        "--mode",
        default="scratch",
        choices=list(TRAINING_MODES),
        help=(
            "scratch: conv encoder from scratch | "
            "bert-ctc: pretrained BERT + CTC head | "
            "neurolm-sft: fine-tune the full NeuroLM model | "
            "neurolm-adapter: HF causal LM + adapter (+ NeuroProcessor)"
        ),
    )
    p_train.add_argument(
        "--pretrained",
        default=None,
        help="checkpoint name/path for the pretrained backbone (BERT or NeuroProcessor)",
    )
    p_train.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="freeze the pretrained encoder and train only adapters and the CTC head",
    )
    p_train.add_argument(
        "--processor",
        default=None,
        choices=["scratch", "pretrained"],
        help="for neurolm-adapter: train the NeuroProcessor from scratch, or use a pretrained one",
    )
    p_train.add_argument(
        "--lm-name", default=None, help="HuggingFace causal LM for the neurolm modes"
    )
    p_train.add_argument(
        "--eeg-tokens", type=int, default=None, help="positional embedding size for EEG frames"
    )
    p_train.add_argument(
        "--frontend-stride",
        type=int,
        default=16,
        help="temporal stride of the convolutional front-end (bert-ctc)",
    )
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument(
        "--max-steps", type=int, default=None, help="stop after N optimizer steps (smoke test)"
    )

    from bci_eeg_to_text.benchmark_cli import add_benchmark_subparser

    add_benchmark_subparser(sub)

    p_eval = sub.add_parser("evaluate", help="evaluate a checkpoint")
    p_eval.add_argument(
        "--split",
        default=None,
        help="split tree to evaluate on (uses its test part); preferred over --cache",
    )
    p_eval.add_argument("--split-part", default=None, help="which part of --split to evaluate on")
    p_eval.add_argument("--cache", default=None, help="directory of pre-extracted .npz trials")
    p_eval.add_argument("--cache-dir", default=None, help="where extracted trials are cached")
    p_eval.add_argument("--decimate", type=int, default=8)
    p_eval.add_argument("--out", default=None, help="used to locate the cache directory")
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--device", default=None)
    p_eval.add_argument("--n-samples", type=int, default=8000)
    p_eval.add_argument("--beam-width", type=int, default=1, help="1 means greedy")
    return parser


def _load_caches(caches: list[Path], n_samples: int):
    """Build a dataset and tokenizer from explicit cache files."""
    import numpy as np

    from bci_eeg_to_text.data import TrialDataset
    from bci_eeg_to_text.tokenizer import CharTokenizer

    if not caches:
        raise SystemExit("no trials to evaluate on")
    texts: list[str] = []
    n_channels: int | None = None
    for path in caches:
        with np.load(path, allow_pickle=True) as data:
            texts.extend(str(t) for t in data["texts"])
            if n_channels is None:
                n_channels = int(data["n_channels"])
    tokenizer = CharTokenizer.build(texts)
    return TrialDataset(caches, tokenizer, n_samples=n_samples), tokenizer, int(n_channels or 0)


def _extract_split_part(tree, part: str, args: argparse.Namespace) -> tuple[list[Path], list[str]]:
    """Extract one part of a split tree into its cache directory."""
    from bci_eeg_to_text.splitdata import extract_part

    cache_dir = Path(args.cache_dir or (Path(args.out or ".") / "cache"))
    return extract_part(tree, part, cache_dir / part, decimate=args.decimate)


def _evaluate_leakage(split, sources, participants, args):
    """Build the leakage matrix for a computed split.

    Subject, text, sex and run axes are evaluated together so the report names
    exactly which axis leaks. Chisco contributes no paragraph/article structure,
    so text modes above ``row`` are only meaningful for ChineseEEG.
    """
    from bci_eeg_to_text.leakage import evaluate_matrix, summarize_coverage

    # At run level the entries are "subject/run", so the subject axis must look
    # at the subject half while the run axis uses the whole entry. Treating the
    # full entry as a subject made one participant look like eighteen.
    by_id = {p.subject_id: p for p in participants}
    subject_parts: dict[str, list[str]] = {}
    run_parts: dict[str, list[str]] = {}
    for part, entries in split.parts.items():
        if split.subject_level:
            subject_parts[part] = list(entries)
        else:
            run_parts[part] = list(entries)
            seen: dict[str, None] = {}
            for entry in entries:
                seen.setdefault(entry.split("/")[0], None)
            subject_parts[part] = list(seen)

    units = _collect_text_units(sources)
    if units:
        print(f"text coverage: {summarize_coverage(units, args.leakage)}")
    text_parts: dict[str, list] = {}
    if units:
        from bci_eeg_to_text.leakage import assign_units_to_parts

        ratio_keys = {part: 1 for part in split.parts}
        assigned = assign_units_to_parts(
            units, mode=args.leakage, ratios=ratio_keys, seed=args.seed
        )
        text_parts = {part: list(group) for part, group in assigned.items()}

    # A single-subject split has nothing to balance, so reporting sex imbalance
    # there would be noise rather than signal.
    single = args.subject_mode == "one_subject"
    return evaluate_matrix(
        subject_parts=subject_parts,
        text_parts=text_parts,
        participants=by_id,
        run_parts=run_parts,
        subject_mode=args.subject_mode,
        text_mode=args.leakage,
        check_sex=not single,
    )


def _collect_text_units(sources) -> list:
    """Every labelled row across the corpora that have source documents.

    Rows are located inside the novels by string search, so the caller can
    report how much structure was recovered rather than implying full coverage.
    """
    from bci_eeg_to_text.chineseeeg import read_segmented_novel
    from bci_eeg_to_text.leakage import units_from_labels
    from bci_eeg_to_text.textstructure import (
        TextStructureError,
        find_source_documents,
        label_rows,
        read_document,
    )

    units: list = []
    for source in sources:
        if source.name not in DOCUMENT_DATASETS:
            # Chisco stimuli are standalone sentences with no surrounding
            # document, so only row-level identity exists there.
            continue
        for novel, path in find_source_documents(source.root()).items():
            try:
                document = read_document(path, novel)
            except TextStructureError:
                continue
            # Every participant read the SAME novel, so rows are collected per
            # document, not per (subject, run), or each sentence is counted once
            # per participant and totals inflate ~10x.
            seen_files: set[str] = set()
            for run in source.runs():
                if novel not in (run.session or ""):
                    continue
                paths = [p for p in source.novel_paths(run) if "_display" not in p.name]
                if not paths or paths[0].name in seen_files:
                    continue
                seen_files.add(paths[0].name)
                try:
                    texts = read_segmented_novel(paths[0])
                except Exception:  # noqa: BLE001
                    continue
                # Each run file restarts partway through the novel, so the
                # search cursor must reset or later files locate nothing.
                document.reset_cursor()
                units.extend(units_from_labels(label_rows(texts, document), source.name))
    return units


def _data_from_split(args: argparse.Namespace):
    """Build datasets from a materialised split tree.

    Training reads the split, so subject-level and run-level splits behave
    identically here: the difference is only what the links point at.
    """
    from bci_eeg_to_text.splitdata import BuildError, build_training_data
    from bci_eeg_to_text.splittree import (
        SplitTreeError,
        check_no_leakage,
        load_split_tree,
        summarise,
    )

    try:
        tree = load_split_tree(args.split)
        # Leakage is an error, not a warning: a trial shared between train and
        # test inflates scores while looking like a good model.
        check_no_leakage(tree)
    except SplitTreeError as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(summarise(tree))
    for warning in tree.warnings:
        print(f"warning: {warning}", file=sys.stderr)

    cache_dir = Path(args.cache_dir or (Path(args.out) / "cache"))
    try:
        built = build_training_data(
            tree,
            n_samples=args.n_samples,
            cache_dir=cache_dir,
            decimate=args.decimate,
        )
    except BuildError as exc:
        raise SystemExit(f"error: {exc}") from exc
    return (
        built.train,
        built.valid,
        built.tokenizer,
        built.n_channels,
        built.valid_part,
    )


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

    # Validate arguments before touching the filesystem: a malformed --ratios
    # must fail the same way whether or not the dataset happens to be mounted.
    ratios = _parse_ratios(args.ratios, args.dataset)
    config = _data_config(args)

    names = _resolve_dataset_names(args.dataset)
    sources = [get_source(name, **_source_kwargs(config, name)) for name in names]
    participants = subjects_for_split(sources)
    if not participants:
        print("no participants found; is the dataset mounted?", file=sys.stderr)
        return 1

    chosen = None
    if args.subject_mode == "one_subject":
        # Pick one participant at random and split only its runs. Sex balance is
        # deliberately not considered: with one subject there is nothing to
        # balance, and the point is a within-participant experiment.
        from bci_eeg_to_text.leakage import pick_one_subject

        try:
            chosen = pick_one_subject(participants, text_mode=args.leakage, seed=args.seed)
        except LeakageError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        participants = [chosen]
        args.level = "run"
        print(
            f"subject mode one_subject: using {chosen.subject_id} only; "
            f"all parts share this participant, so scores are within-participant "
            f"and not a generalisation estimate"
        )

    runs_per_subject = None
    if args.level == "run":
        wanted = {p.subject_id for p in participants}
        runs_per_subject = {}
        for source in sources:
            for run in source.runs():
                key = f"{source.name}:{run.subject_id}"
                if key not in wanted:
                    continue
                # Session-qualify the label: a run number repeats across
                # sessions (sub-04 has run-01 in two of them), so a bare
                # "run-01" denotes two recordings and lands in two parts.
                label = f"{run.session}/{run.run}" if run.session else run.run
                runs_per_subject.setdefault(key, []).append(label)

    split = make_splits(
        participants,
        ratios=ratios,
        level=args.level,
        seed=args.seed,
        runs_per_subject=runs_per_subject,
    )
    validate_subject_disjoint(split, participants)

    # Evaluate every leakage axis and print the whole matrix. It is printed
    # unconditionally: silence would be indistinguishable from "no leakage",
    # which is precisely the confusion the matrix exists to remove.
    matrix = _evaluate_leakage(split, sources, participants, args)
    print(matrix.describe())

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

    if args.link:
        from bci_eeg_to_text.dataconfig import DataAccessError, materialise_split

        corpus_paths = {
            source.name: config.dataset_path(_corpus_dirname(source.name)) for source in sources
        }
        try:
            result = materialise_split(
                split,
                data=config,
                corpus_paths=corpus_paths,
                cwd=Path.cwd(),
                leakage_mode=args.leakage,
                leakage_violations=len(matrix.text.violations),
                subject_mode=args.subject_mode,
            )
        except DataAccessError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        where = "inside the dataset root" if config.assume_writable else "in the current directory"
        print(f"\nlinked split tree {where}: {result.path}")
        print(f"  {result.n_links} symlinks (relative targets, no data copied)")
        for part, count in sorted(result.parts.items()):
            print(f"  {part}: {count}")
        if result.skipped:
            print(f"  {len(result.skipped)} entries skipped (no local files)", file=sys.stderr)
        print(f"  manifest: {result.manifest_path}")
        print("\nBrowse it with, e.g.:")
        print(f"  ls -l {result.path}/train | head")

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

    config = _data_config(args)
    names = _resolve_dataset_names(args.dataset)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    skipped: list[tuple[str, str]] = []
    for name in names:
        source = get_source(name, **_source_kwargs(config, name))
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


def _corpus_dirname(name: str) -> str:
    """OpenNeuro directory name for a registered source."""
    return {"chisco": "ds005170", "chineseeeg": "ds004952"}.get(name, name)


def _data_config(args: argparse.Namespace) -> DataConfig:
    """Resolve the dataset location from CLI flags and the environment."""
    return DataConfig.from_env(
        root=getattr(args, "data_root", None),
        subdir=getattr(args, "data_subdir", None),
        assume_writable=getattr(args, "data_writable", None),
    )


def _source_kwargs(config: DataConfig, name: str) -> dict[str, object]:
    """Keyword arguments needed to build a source for ``name``."""
    return {"data_root": config.dataset_path(_corpus_dirname(name))}


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

    from bci_eeg_to_text.backbones import build_ctc_model
    from bci_eeg_to_text.train import (
        TrainConfig,
        evaluate,
        make_batches,
        resolve_device,
        train_one_epoch,
    )

    if not args.split and not args.cache:
        print(
            "error: provide --split <tree> (preferred) or --cache <dir>.\n"
            "Create a split first:\n"
            "  bci-eeg-to-text split --dataset chisco --link\n"
            "  bci-eeg-to-text train --split splits --out runs/exp1",
            file=sys.stderr,
        )
        return 2

    device = resolve_device(args.device)
    if args.split:
        dataset, valid_dataset, tokenizer, n_channels, valid_part = _data_from_split(args)
    else:
        dataset, tokenizer, n_channels = _load_dataset(Path(args.cache), args.n_samples)
        valid_dataset, valid_part = None, None
    print(
        f"device={device} mode={args.mode} trials={len(dataset)} "
        f"channels={n_channels} classes={len(tokenizer)}"
    )
    if valid_dataset is None:
        print(
            "warning: no validation split; falling back to a holdout from the "
            "training data. Use --split for a participant-level evaluation.",
            file=sys.stderr,
        )

    torch.manual_seed(args.seed)
    extra: dict[str, object] = {"frontend_stride": args.frontend_stride}
    if args.eeg_tokens:
        extra["eeg_tokens"] = args.eeg_tokens
    if args.processor:
        extra["processor"] = args.processor
    if args.lm_name:
        extra["lm_name"] = args.lm_name

    model = build_ctc_model(
        args.mode,
        n_channels=n_channels,
        n_classes=len(tokenizer),
        d_model=args.d_model,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        extra=extra,
    ).to(device)

    summary = model.parameter_summary()
    print(
        f"parameters: total={summary['total']:,.0f} "
        f"trainable={summary['trainable']:,.0f} "
        f"({summary['trainable_fraction']:.1%}) frozen={summary['frozen']:,.0f}"
    )
    if args.freeze_backbone and summary["trainable_fraction"] >= 1.0:
        print("warning: --freeze-backbone had no effect for this mode", file=sys.stderr)

    trials = [dataset[i] for i in range(len(dataset))]
    if valid_dataset is not None:
        # A real held-out part: no trial is shared with training.
        train_batches = make_batches(trials, args.batch_size, shuffle=True, seed=args.seed)
        valid_batches = make_batches(
            [valid_dataset[i] for i in range(len(valid_dataset))], args.batch_size
        )
        print(f"validation part: {valid_part} ({len(valid_dataset)} trials)")
    else:
        split = max(1, int(len(trials) * 0.9))
        train_batches = make_batches(trials[:split], args.batch_size, shuffle=True, seed=args.seed)
        valid_batches = make_batches(trials[split:], args.batch_size)

    config = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
    # Optimise only what is trainable; passing frozen weights to AdamW wastes
    # memory on their state and can silently unfreeze them via weight decay.
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
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
                "mode": args.mode,
                "backbone_kwargs": {
                    "d_model": args.d_model,
                    "pretrained": args.pretrained,
                    "freeze_backbone": args.freeze_backbone,
                    "extra": extra,
                },
                "n_channels": n_channels,
                "n_classes": len(tokenizer),
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

    from bci_eeg_to_text.backbones import build_ctc_model
    from bci_eeg_to_text.decode import (
        beam_search_decode,
        character_error_rate,
        sentence_accuracy,
    )
    from bci_eeg_to_text.train import evaluate, make_batches, move_batch_to_device, resolve_device

    if not args.split and not args.cache:
        print(
            "error: provide --split <tree> (preferred) or --cache <dir>",
            file=sys.stderr,
        )
        return 2

    device = resolve_device(args.device)
    if args.split:
        # Evaluate on the held-out part of the same split used for training.
        from bci_eeg_to_text.splittree import (
            SplitTreeError,
            check_no_leakage,
            load_split_tree,
            summarise,
        )

        try:
            tree = load_split_tree(args.split)
            check_no_leakage(tree)
        except SplitTreeError as exc:
            raise SystemExit(f"error: {exc}") from exc
        print(summarise(tree))
        part = args.split_part or ("test" if "test" in tree.parts else None)
        if part is None or not tree.parts.get(part):
            raise SystemExit("error: --split has no test part to evaluate on")
        eval_paths, _ = _extract_split_part(tree, part, args)
        dataset, tokenizer, n_channels = _load_caches(eval_paths, args.n_samples)
        print(f"evaluating on part {part!r} ({len(dataset)} trials)")
    else:
        dataset, tokenizer, n_channels = _load_dataset(Path(args.cache), args.n_samples)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # Rebuild exactly the architecture that was trained; the mode and backbone
    # options are stored in the checkpoint for this reason.
    model = build_ctc_model(
        checkpoint.get("mode", "scratch"),
        n_channels=int(checkpoint.get("n_channels", n_channels)),
        n_classes=int(checkpoint.get("n_classes", len(tokenizer))),
        **checkpoint.get("backbone_kwargs", {"d_model": 256}),
    ).to(device)
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
        # `benchmark` carries its own subcommand; dispatch through the args.
        "benchmark": lambda a: a.func(a),
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
