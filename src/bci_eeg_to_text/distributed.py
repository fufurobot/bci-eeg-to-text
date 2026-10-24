"""Training backends: CPU, local GPU, and passwordless-SSH clusters.

Two decisions are validated before any expensive work starts:

1. **Where to run.** ``cpu``, ``gpu`` or ``ssh``. A request for ``gpu`` fails
   loudly when no CUDA device exists rather than quietly running on CPU, which
   here is roughly two orders of magnitude slower -- a silent fallback turns a
   ten-minute job into hours and looks like a modelling failure.
2. **How to distribute.** ``dataparallel`` replicates the whole model into every
   device, so it cannot exceed one device's memory. ``fsdp`` shards parameters,
   gradients and optimizer state across ranks, so a model larger than one GPU
   can train; it needs one process per device.

Clusters are reached over **passwordless SSH**, and topologies may be
**multi-hop** (``jump1|jump2|worker``), because machines in a lab are usually
reachable only through a bastion. Jump hosts are chained with ``-J`` and every
hop gets ``BatchMode=yes`` so an unreachable host fails fast instead of hanging
on a prompt that nobody can answer.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

#: Supported training backends.
TRAINING_MODES = ("cpu", "gpu", "ssh")

#: Supported distribution strategies.
STRATEGIES = ("dataparallel", "fsdp")


class ModeUnavailable(RuntimeError):
    """A requested backend cannot run on this machine or topology."""


class ClusterUnreachable(RuntimeError):
    """One or more hops of an SSH topology could not be reached."""


@dataclass(frozen=True)
class TrainingMode:
    """A validated choice of where training runs."""

    kind: str
    n_devices: int = 0
    hosts: tuple[str, ...] = ()
    requires_gpu: bool = False


def resolve_mode(
    kind: str,
    *,
    cuda_available: bool,
    n_devices: int | None = None,
    hosts: list[str] | None = None,
) -> TrainingMode:
    """Validate a backend request, raising instead of degrading silently."""
    if kind not in TRAINING_MODES:
        raise ValueError(
            f"unknown training mode {kind!r}; expected one of {', '.join(TRAINING_MODES)}"
        )
    if kind == "cpu":
        return TrainingMode(kind="cpu", requires_gpu=False)

    if kind == "gpu":
        if not cuda_available:
            raise ModeUnavailable(
                "CUDA was requested (--training-mode gpu) but no CUDA device is "
                "available. Refusing to fall back to CPU, which is ~125x slower. "
                "Use --training-mode cpu if that cost is intended."
            )
        return TrainingMode(kind="gpu", n_devices=int(n_devices or 1), requires_gpu=True)

    if not hosts:
        raise ValueError(
            "training mode 'ssh' needs at least one host; pass --hosts, e.g. "
            "--hosts jump:22|worker:4"
        )
    return TrainingMode(kind="ssh", hosts=tuple(hosts), requires_gpu=True)


@dataclass(frozen=True)
class DistributedPlan:
    """How a strategy maps onto processes and memory."""

    strategy: str
    n_processes: int
    shards_parameters: bool


def plan_distributed(strategy: str, *, n_devices: int) -> DistributedPlan:
    """Translate a strategy name into process count and sharding behaviour."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {', '.join(STRATEGIES)}")
    if strategy == "dataparallel":
        # One process holds the model and mirrors it onto each device.
        return DistributedPlan(strategy, n_processes=1, shards_parameters=False)
    if n_devices < 2:
        raise ValueError(
            "fsdp needs at least two devices to shard across; use dataparallel for a single device"
        )
    return DistributedPlan(strategy, n_processes=n_devices, shards_parameters=True)


def parse_topology(spec: str) -> list[str]:
    """Parse ``jump1:22|jump2:22|worker:4`` into its hop list.

    The final hop is the worker; preceding hops are jump hosts. ``|`` separates
    hops so that a flat comma-separated list stays unambiguous.
    """
    if not spec or not spec.strip():
        raise ValueError("empty topology; expected e.g. 'worker:4' or 'jump:22|worker:4'")
    hops = [h.strip() for h in spec.split("|") if h.strip()]
    if not hops:
        raise ValueError("empty topology; expected at least one host")
    for hop in hops:
        host, _, count = hop.partition(":")
        if not host:
            raise ValueError(f"malformed hop {hop!r}: expected host:count")
        if count:
            try:
                n = int(count)
            except ValueError as exc:
                raise ValueError(f"malformed hop {hop!r}: count must be an integer") from exc
            if n <= 0:
                raise ValueError(f"malformed hop {hop!r}: count must be positive")
    return hops


def _host_of(hop: str) -> str:
    return hop.partition(":")[0]


def worker_count(hops: list[str]) -> int:
    """Number of workers, taken from the last hop."""
    last = hops[-1]
    _, _, count = last.partition(":")
    return int(count) if count else 1


def build_ssh_command(
    hops: list[str],
    *,
    remote_cmd: str,
    identity: Path | None = None,
    user: str | None = None,
) -> list[str]:
    """Build an ``ssh`` argv that traverses every hop.

    ``-J`` takes a comma-separated jump chain, so a three-hop topology becomes
    ``-J jump1,jump2`` for the final worker.
    """
    if not hops:
        raise ValueError("no hops given")
    worker = hops[-1]
    argv = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=10",
    ]
    if identity is not None:
        argv += ["-i", str(identity)]
    if len(hops) > 1:
        argv += ["-J", ",".join(_host_of(h) for h in hops[:-1])]
    target = worker if "@" in worker else (f"{user}@{worker}" if user else worker)
    argv += [target, remote_cmd]
    return argv


def build_launch_command(
    *,
    plan: DistributedPlan,
    script: Path,
    extra: list[str],
    python: str | None = None,
) -> list[str]:
    """Local launch: ``torchrun`` for sharded strategies, plain python otherwise."""
    py = python or "python"
    if plan.n_processes > 1:
        # Prefer the torchrun console script, but fall back to the module form:
        # torchrun lives in the venv's bin/ and is often absent from PATH, which
        # would otherwise make FSDP look unavailable on a perfectly good install.
        launcher = "torchrun" if shutil.which("torchrun") else f"{py} -m torch.distributed.run"
        return [
            *launcher.split(),
            f"--nproc_per_node={plan.n_processes}",
            str(script),
            *extra,
        ]
    return [py, str(script), *extra]


def build_ssh_launch(
    *,
    chain: list[str],
    plan: DistributedPlan,
    script: Path,
    extra: list[str],
    identity: Path | None = None,
    python: str | None = None,
) -> list[str]:
    """Wrap a launch command so it runs on the final host of the chain."""
    inner = build_launch_command(plan=plan, script=script, extra=extra, python=python)
    return build_ssh_command(chain, remote_cmd=" ".join(inner), identity=identity)


@dataclass(frozen=True)
class HostStatus:
    """Whether one hop answered."""

    host: str
    reachable: bool
    reason: str = ""


def probe_cluster(
    chains: list[list[str]] | list[str],
    *,
    runner: Callable[[str], bool] | None = None,
    identity: Path | None = None,
    timeout: int = 10,
) -> list[HostStatus]:
    """Check every hop of every topology before launching anything."""
    statuses: list[HostStatus] = []
    probe = runner or (lambda hop: _probe_hop(hop, identity=identity, timeout=timeout))
    for chain in chains:
        hops = [chain] if isinstance(chain, str) else chain
        for hop in hops:
            try:
                ok = bool(probe(hop))
            except Exception as exc:  # noqa: BLE001 - reported per host
                statuses.append(HostStatus(hop, False, f"{type(exc).__name__}: {exc}"))
                continue
            statuses.append(HostStatus(hop, ok, "" if ok else "ssh probe failed (passwordless?)"))
    return statuses


def _probe_hop(hop: str, *, identity: Path | None, timeout: int) -> bool:
    import subprocess

    argv = build_ssh_command([hop], remote_cmd="true", identity=identity)
    # Replace the default timeout with the caller's.
    argv[argv.index("ConnectTimeout=10")] = f"ConnectTimeout={timeout}"
    try:
        return subprocess.run(argv, capture_output=True, timeout=timeout + 5).returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def require_reachable(statuses: list[HostStatus]) -> None:
    """Abort before launch if any hop is down, naming every failure."""
    bad = [s for s in statuses if not s.reachable]
    if bad:
        detail = "; ".join(f"{s.host} ({s.reason})" for s in bad)
        raise ClusterUnreachable(f"unreachable hosts: {detail}")


@dataclass
class AcceleratorConfig:
    """Everything a training entrypoint needs about placement."""

    mode: TrainingMode
    plan: DistributedPlan
    backend: str = "ddp"
    extra: dict[str, object] = field(default_factory=dict)
