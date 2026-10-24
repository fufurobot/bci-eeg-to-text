"""Two unified entrypoints: ``bci dataset`` and ``bci experiment``.

The CLI had grown one command per concern (prepare, split, train, evaluate,
inspect, benchmark), each with its own dataset flags. As corpora and tasks
multiply that approach duplicates flags and drifts. These entrypoints group the
work by what the user is managing:

* ``bci dataset list | info NAME | prepare NAME`` -- corpora.
* ``bci experiment list | run NAME`` -- benchmark rows and training runs.

Both resolve names through :mod:`bci_eeg_to_text.registry`, so a new corpus or a
future EEG task appears here without any change to this file.
"""

from __future__ import annotations

import argparse
import sys

from bci_eeg_to_text.registry import (
    RegistryError,
    get_dataset,
    get_experiment,
    list_datasets,
    list_experiments,
)

#: Training backends, kept in step with :mod:`bci_eeg_to_text.distributed`.
TRAINING_MODES = ("cpu", "gpu", "ssh")
STRATEGIES = ("dataparallel", "fsdp")


def add_dataset_subparser(sub: argparse._SubParsersAction) -> None:
    """Register ``bci dataset`` and its subcommands."""
    p = sub.add_parser("dataset", help="list, inspect and prepare corpora")
    ds = p.add_subparsers(dest="dataset_command", required=True)

    p_list = ds.add_parser("list", help="list known corpora")
    p_list.add_argument("--task", default=None, help="filter by task")
    p_list.set_defaults(func=cmd_dataset_list)

    p_info = ds.add_parser("info", help="describe one corpus")
    p_info.add_argument("name")
    p_info.set_defaults(func=cmd_dataset_info)

    p_prep = ds.add_parser("prepare", help="stage a split tree for one corpus")
    p_prep.add_argument("name")
    p_prep.add_argument("--out", default="splits")
    p_prep.add_argument("--leakage", default="row")
    p_prep.add_argument("--subject-mode", default="cross_subject")
    p_prep.add_argument("--seed", type=int, default=0)
    p_prep.set_defaults(func=cmd_dataset_prepare)


def add_experiment_subparser(sub: argparse._SubParsersAction) -> None:
    """Register ``bci experiment`` and its subcommands."""
    p = sub.add_parser("experiment", help="list and run experiments")
    ex = p.add_subparsers(dest="experiment_command", required=True)

    p_list = ex.add_parser("list", help="list known experiments")
    p_list.add_argument("--task", default="eeg2text")
    p_list.set_defaults(func=cmd_experiment_list)

    p_run = ex.add_parser("run", help="run one experiment")
    p_run.add_argument("name")
    p_run.add_argument("--dataset", action="append", required=True, help="corpus name; repeatable")
    p_run.add_argument(
        "--training-mode",
        default="cpu",
        choices=list(TRAINING_MODES),
        help="cpu | gpu | ssh (passwordless SSH cluster)",
    )
    p_run.add_argument(
        "--strategy",
        default="dataparallel",
        choices=list(STRATEGIES),
        help="dataparallel replicates per device; fsdp shards (model > 1 GPU)",
    )
    p_run.add_argument(
        "--hosts",
        default=None,
        help=(
            "SSH topology for --training-mode ssh; '|' separates hops, the last "
            "is the worker: e.g. 'jump:22|worker:4'"
        ),
    )
    p_run.add_argument("--identity", default=None, help="SSH private key")
    p_run.add_argument("--smoke", action="store_true", help="tiny slice, 5-fold 3-1-1")
    p_run.add_argument("--epochs", type=int, default=3)
    p_run.add_argument("--batch-size", type=int, default=8)
    p_run.add_argument("--out", default="runs/experiment")
    p_run.add_argument("--seed", type=int, default=0)
    p_run.set_defaults(func=cmd_experiment_run)


def cmd_dataset_list(args: argparse.Namespace) -> int:
    names = list_datasets(task=args.task)
    if not names:
        print("no datasets registered", file=sys.stderr)
        return 1
    width = max(len(n) for n in names)
    for name in names:
        spec = get_dataset(name)
        print(f"{name:<{width}}  {spec.openneuro_id:<10}  {spec.description}")
    return 0


def cmd_dataset_info(args: argparse.Namespace) -> int:
    try:
        spec = get_dataset(args.name)
    except RegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"name:        {spec.name}")
    print(f"description: {spec.description}")
    print(f"openneuro:   {spec.openneuro_id}")
    print(f"task:        {spec.task}")
    print(f"text unit:   {spec.text_unit}")
    print(f"subdir:      {spec.subdir}")
    for key, value in spec.notes.items():
        print(f"{key + ':':<13}{value}")
    return 0


def cmd_dataset_prepare(args: argparse.Namespace) -> int:
    from bci_eeg_to_text import cli

    try:
        spec = get_dataset(args.name)
    except RegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    # Delegate to the existing, tested split machinery rather than duplicating it.
    forwarded = [
        "split",
        "--dataset",
        args.name,
        "--out",
        args.out,
        "--leakage",
        args.leakage,
        "--subject-mode",
        args.subject_mode,
        "--seed",
        str(args.seed),
    ]
    if spec.task != "eeg2text":
        print(
            f"error: {spec.name} supports task {spec.task!r}, which has no split policy yet",
            file=sys.stderr,
        )
        return 1
    return cli.main(forwarded)


def cmd_experiment_list(args: argparse.Namespace) -> int:
    names = list_experiments(task=args.task)
    if not names:
        print(f"no experiments for task {args.task!r}", file=sys.stderr)
        return 1
    width = max(len(n) for n in names)
    for name in names:
        spec = get_experiment(name)
        marker = "in-process" if spec.trains_in_process else "needs weights"
        print(f"{name:<{width}}  {marker:<13}  {spec.description}")
    return 0


def cmd_experiment_run(args: argparse.Namespace) -> int:
    """Validate placement first, then hand off to the benchmark runner."""
    import shutil

    import torch

    from bci_eeg_to_text import distributed as dist
    from bci_eeg_to_text.registry import RegistryError as _RegistryError

    try:
        spec = get_experiment(args.name)
    except _RegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    hosts = dist.parse_topology(args.hosts) if args.hosts else None
    n_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    try:
        mode = dist.resolve_mode(
            args.training_mode,
            cuda_available=torch.cuda.is_available(),
            n_devices=n_devices,
            hosts=hosts,
        )
        if mode.kind == "ssh":
            from pathlib import Path

            identity = Path(args.identity) if args.identity else None
            statuses = dist.probe_cluster(hosts, identity=identity)
            dist.require_reachable(statuses)
            devices = dist.worker_count(hosts)
        else:
            devices = max(1, n_devices if mode.kind == "gpu" else 1)
        plan = dist.plan_distributed(args.strategy, n_devices=devices)
    except (ValueError, dist.ModeUnavailable, dist.ClusterUnreachable) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    launch = dist.build_launch_command(
        plan=plan,
        script=shutil.os.path.join("scripts", "train_eeg_ctc.py"),
        extra=[
            "--mode",
            "bert-ctc" if spec.name == "conv-bert" else spec.name,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--out",
            f"{args.out}/{spec.name}",
        ],
    )
    print(f"training mode: {mode.kind}  strategy: {plan.strategy}")
    print(f"launch: {' '.join(launch)}")
    if args.smoke:
        from bci_eeg_to_text.benchmark_cli import run_eeg2text_benchmark

        smoke_args = argparse.Namespace(
            model=spec.name if spec.name in ("conv-transformer",) else "all",
            dataset=args.dataset,
            smoke=True,
            data_root=None,
            out=args.out,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
            echo_ckpt=__import__("pathlib").Path("~/ECHO/checkpoints/ECHO.ckpt").expanduser(),
            echo_code=__import__("pathlib").Path("~/ECHO_code/ECHO").expanduser(),
            thought2text_dir=__import__("pathlib").Path("~/Thought2Text").expanduser(),
            hf_models=__import__("pathlib").Path("~/hf-models").expanduser(),
            no_mlflow=True,
        )
        return run_eeg2text_benchmark(smoke_args)
    print(
        "note: a full run needs the training script to accept --mode; the smoke "
        "path is the supported route today",
        file=sys.stderr,
    )
    return 0
