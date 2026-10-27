"""`bci benchmark eeg2text`: compare four EEG-to-text experiments.

The four rows are deliberately different in kind, because that is the comparison
the user asked for:

* ``echo`` -- wythedee/ECHO: FAST encoder, projection, Whisper decoder.
* ``thought2text`` -- an LLM with a ChannelNet EEG encoder.
* ``conv-transformer`` -- ours: conv front-end plus transformer, CTC, one
  forward pass, no autoregressive loop.
* ``neurolm`` -- NeuroLM's NeuralTransformerVQ plus its adapter head.

Every row is produced through the same :mod:`bci_eeg_to_text.benchmark` scoring,
and a row that cannot run is emitted as a labelled skip **with a reason**. That
rule matters most here: three of these experiments depend on weights or code
that may be absent, and a zero would look like a bad model rather than a missing
one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bci_eeg_to_text.benchmark import ModelStatus, format_table

#: Experiment names accepted by ``--model``; ``all`` expands to these.
BENCHMARK_MODELS = (
    "conv-bert",
    "conv-transformer",
    "echo",
    "thought2text",
    "neurolm",
)

DEFAULT_ECHO = Path("/home/fufu/ECHO/checkpoints/ECHO.ckpt")
DEFAULT_ECHO_CODE = Path("/home/fufu/ECHO_code/ECHO")
DEFAULT_T2T = Path("/home/fufu/Thought2Text")
DEFAULT_HF_MODELS = Path("hf-models")


def add_benchmark_subparser(sub: argparse._SubParsersAction) -> None:
    """Register ``benchmark`` and its ``eeg2text`` subcommand."""
    p_bench = sub.add_parser("benchmark", help="compare EEG-to-text experiments")
    bench_sub = p_bench.add_subparsers(dest="bench_command", required=True)
    p = bench_sub.add_parser("eeg2text", help="EEG -> text benchmark")

    p.add_argument(
        "--model",
        default="all",
        choices=[*BENCHMARK_MODELS, "all"],
        help="experiment to run; 'all' runs the four named ones",
    )
    p.add_argument(
        "--dataset",
        action="append",
        default=None,
        metavar="NAME",
        help="dataset to benchmark; repeat for several (chineseeeg, chisco)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "tiny sample: 10 runs holding 5 texts x 2 runs from one subject, "
            "with 5-fold 3-1-1 cross-validation"
        ),
    )
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("runs/benchmark"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=3, help="training epochs per fold")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--echo-ckpt", type=Path, default=DEFAULT_ECHO)
    p.add_argument("--echo-code", type=Path, default=DEFAULT_ECHO_CODE)
    p.add_argument("--thought2text-dir", type=Path, default=DEFAULT_T2T)
    p.add_argument("--hf-models", type=Path, default=DEFAULT_HF_MODELS)
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument(
        "--max-temp",
        type=float,
        default=80.0,
        help="GPU ceiling in Celsius (hard limit); this machine's fan is broken",
    )
    p.add_argument(
        "--max-cpu-temp",
        type=float,
        default=88.0,
        help=(
            "CPU ceiling in Celsius; higher than the GPU ceiling because the "
            "external cooling cannot hold the package lower. Lower it when "
            "better fans are fitted."
        ),
    )
    p.add_argument(
        "--thermal-interval",
        type=float,
        default=10.0,
        help="seconds between temperature samples during a run",
    )
    p.set_defaults(func=run_eeg2text_benchmark)


def build_parser() -> argparse.ArgumentParser:
    """The full ``bci`` parser, so benchmark wiring is tested as shipped.

    Reusing the real parser rather than a hand-built subset means a benchmark
    flag cannot exist here but be missing from the installed command.
    """
    from bci_eeg_to_text.cli import build_parser as build_cli_parser

    return build_cli_parser()


def collect_statuses(
    model: str,
    *,
    echo_ckpt: Path = DEFAULT_ECHO,
    echo_code: Path = DEFAULT_ECHO_CODE,
    thought2text_dir: Path = DEFAULT_T2T,
    hf_models: Path = DEFAULT_HF_MODELS,
) -> list[ModelStatus]:
    """One status per requested experiment, always four for ``all``."""
    names = BENCHMARK_MODELS if model == "all" else (model,)
    return [
        probe_special(
            name,
            echo_ckpt=echo_ckpt,
            echo_code=echo_code,
            thought2text_dir=thought2text_dir,
            hf_models=hf_models,
        )
        for name in names
    ]


def probe_special(
    name: str,
    *,
    echo_ckpt: Path,
    echo_code: Path,
    thought2text_dir: Path,
    hf_models: Path,
) -> ModelStatus:
    """Probe a benchmark row, using richer checks than the base registry."""
    if name == "echo":
        if not echo_ckpt.exists():
            return ModelStatus(
                name, False, reason=f"checkpoint missing: {echo_ckpt}", kind="seq2seq"
            )
        if not (echo_code / "FAST").exists():
            return ModelStatus(
                name,
                False,
                reason=f"ECHO code missing at {echo_code} (expected FAST/EEG2Text)",
                kind="seq2seq",
            )
        return ModelStatus(
            name,
            True,
            kind="seq2seq",
            reason="FAST+Whisper checkpoint and code present; fine-tune connect+whisper",
        )

    if name == "thought2text":
        if not thought2text_dir.exists():
            return ModelStatus(
                name, False, reason=f"not found: {thought2text_dir}", kind="causal-lm"
            )
        return ModelStatus(
            name,
            False,
            reason=(
                "code present but needs a base LLM (Mistral-7B) plus stage-2 LoRA "
                "adapters; neither is downloaded"
            ),
            kind="causal-lm",
        )

    if name == "conv-bert":
        return ModelStatus(
            name, True, kind="ctc", reason="conv + BERT encoder trained in-process with CTC"
        )

    if name == "conv-transformer":
        return ModelStatus(
            name, True, kind="ctc", reason="trained in-process with CTC, one forward pass"
        )

    if name == "neurolm":
        candidates = [hf_models, Path("hf-models"), Path("/home/fufu/hf-models")]
        found = next((c for c in candidates if c.exists() and any(c.iterdir())), None)
        if found is None:
            return ModelStatus(
                name,
                False,
                reason="NeuroLM checkpoint not downloaded (expected ~30 GB under hf-models)",
                kind="ctc",
            )
        return ModelStatus(name, True, kind="ctc", reason=f"checkpoint at {found}")

    raise KeyError(f"unknown model {name!r}")


def run_eeg2text_benchmark(args: argparse.Namespace) -> int:
    """Execute the benchmark, printing one table row per experiment."""
    from bci_eeg_to_text.benchmark_runner import execute

    datasets = args.dataset or ["chineseeeg"]
    return execute(
        model=args.model,
        datasets=datasets,
        smoke=args.smoke,
        data_root=args.data_root,
        out=args.out,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        echo_ckpt=args.echo_ckpt,
        echo_code=args.echo_code,
        thought2text_dir=args.thought2text_dir,
        hf_models=args.hf_models,
        use_mlflow=not args.no_mlflow,
        max_temp=args.max_temp,
        max_cpu_temp=args.max_cpu_temp,
        thermal_interval=args.thermal_interval,
        printer=print,
        table=format_table,
    )
