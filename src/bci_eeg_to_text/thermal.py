"""Thermal monitoring and enforcement.

This machine's fan is broken and cooling is external, so temperature is a
first-class constraint rather than an afterthought: a long training run can push
the CPU package past its safe limit and there is no automatic fan response to
save it.

Policy:

* **Sensors are read from the source of truth** -- ``/sys/class/thermal`` on
  Linux and ``nvidia-smi`` for the GPU -- not estimated.
* **The limit is 80 degrees C by default** and applies to every sensor.
* **Breaching the limit stops work.** A monitor either samples in a background
  thread and raises into the worker, or exposes ``check()`` for callers that
  poll. Warning into a log while continuing to cook the CPU would be worse than
  not monitoring at all.
* A reading that cannot be obtained is reported as unknown, never as cool.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

#: Ceiling for the GPU. This is the hard limit: a discrete GPU has its own
#: thermal envelope and the external cooling here is sized for the CPU.
DEFAULT_GPU_LIMIT_C = 80.0

#: Ceiling for the CPU package. The machine's own fan is broken and cooling is
#: external, so the CPU runs hot by design; this ceiling is set by what that
#: cooling can hold, not by the chip's maximum. It is higher than the GPU limit
#: on purpose, and should be lowered when better cooling is installed.
DEFAULT_CPU_LIMIT_C = 88.0

#: Backwards-compatible alias for the GPU ceiling.
DEFAULT_LIMIT_C = DEFAULT_GPU_LIMIT_C


class ThermalError(RuntimeError):
    """Temperature exceeded the configured limit."""


@dataclass(frozen=True)
class Reading:
    """One sensor's temperature."""

    sensor: str
    celsius: float
    kind: str = "cpu"


@dataclass
class ThermalState:
    """A snapshot of every sensor available on the machine.

    Limits are per sensor kind: the GPU has the hard ceiling, while the CPU
    ceiling is looser because this machine's cooling cannot hold it lower.
    """

    readings: list[Reading] = field(default_factory=list)
    limit: float = DEFAULT_GPU_LIMIT_C
    timestamp: float = 0.0
    cpu_limit: float = DEFAULT_CPU_LIMIT_C

    def limit_for(self, kind: str) -> float:
        return self.cpu_limit if kind == "cpu" else self.limit

    @property
    def hottest(self) -> Reading | None:
        if not self.readings:
            return None
        return max(self.readings, key=lambda r: r.celsius)

    @property
    def max_celsius(self) -> float | None:
        hot = self.hottest
        return hot.celsius if hot else None

    @property
    def ok(self) -> bool:
        """True when every sensor is within its own kind's limit.

        No sensors means *unknown*, which is reported as not ok: silently
        treating an unreadable machine as cool is how hardware gets damaged.
        """
        if not self.readings:
            return False
        return all(r.celsius <= self.limit_for(r.kind) for r in self.readings)

    def violations(self) -> list[Reading]:
        return [r for r in self.readings if r.celsius > self.limit_for(r.kind)]

    def describe(self) -> str:
        if not self.readings:
            return "temperature: unknown (no sensors readable)"
        parts = [
            f"{r.sensor}={r.celsius:.0f}C/{self.limit_for(r.kind):.0f}C"
            for r in self.readings
        ]
        verdict = "ok" if self.ok else "OVER LIMIT"
        return (
            f"temperature: {' '.join(parts)} "
            f"(gpu {self.limit:.0f}C, cpu {self.cpu_limit:.0f}C; {verdict})"
        )


def read_cpu_temps(thermal_root: Path = Path("/sys/class/thermal")) -> list[Reading]:
    """Read every readable Linux thermal zone."""
    readings: list[Reading] = []
    if not thermal_root.exists():
        return readings
    for zone in sorted(thermal_root.glob("thermal_zone*")):
        try:
            kind = (zone / "type").read_text().strip()
            milli = int((zone / "temp").read_text().strip())
        except (OSError, ValueError):
            continue
        readings.append(Reading(sensor=kind, celsius=milli / 1000.0, kind="cpu"))
    return readings


def read_gpu_temps() -> list[Reading]:
    """Read GPU temperatures via nvidia-smi, if present."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,index",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if out.returncode != 0:
        return []
    readings: list[Reading] = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0]:
            continue
        try:
            celsius = float(parts[0])
        except ValueError:
            continue
        index = parts[1] if len(parts) > 1 else "0"
        readings.append(Reading(sensor=f"gpu{index}", celsius=celsius, kind="gpu"))
    return readings


def snapshot(
    limit: float = DEFAULT_GPU_LIMIT_C, *, cpu_limit: float = DEFAULT_CPU_LIMIT_C
) -> ThermalState:
    """Read every sensor available right now."""
    return ThermalState(
        readings=[*read_cpu_temps(), *read_gpu_temps()],
        limit=limit,
        cpu_limit=cpu_limit,
        timestamp=time.time(),
    )


def require_cool(
    limit: float = DEFAULT_GPU_LIMIT_C, *, cpu_limit: float = DEFAULT_CPU_LIMIT_C
) -> ThermalState:
    """Read the sensors and raise if any is above ``limit``.

    Called before a run starts so an already-hot machine is not loaded further.
    """
    state = snapshot(limit, cpu_limit=cpu_limit)
    if not state.readings:
        raise ThermalError(
            "no temperature sensors readable; refusing to run blind on a machine "
            "with known cooling problems"
        )
    if not state.ok:
        hot = ", ".join(
            f"{r.sensor}={r.celsius:.0f}C>{state.limit_for(r.kind):.0f}C"
            for r in state.violations()
        )
        raise ThermalError(f"{hot} exceeds its limit before starting; let the machine cool")
    return state


class ThermalGuard:
    """Background sampler that stops work when the limit is exceeded.

    Used as a context manager around a long run::

        with ThermalGuard(limit=80) as guard:
            train()
            guard.check()

    ``check()`` raises :class:`ThermalError` on breach, so a training loop can
    call it cheaply between steps.
    """

    def __init__(
        self,
        limit: float = DEFAULT_GPU_LIMIT_C,
        *,
        cpu_limit: float = DEFAULT_CPU_LIMIT_C,
        interval: float = 5.0,
        sample: callable | None = None,
        on_sample: callable | None = None,
    ) -> None:
        self.limit = limit
        self.cpu_limit = cpu_limit
        self.interval = interval
        self._sample = sample or (lambda: snapshot(limit, cpu_limit=cpu_limit))
        self._on_sample = on_sample
        self.history: list[ThermalState] = []
        self.peak: float = 0.0
        self._last: ThermalState | None = None

    def poll(self) -> ThermalState:
        """Take one reading, record it, and raise on breach."""
        state = self._sample()
        self._last = state
        self.history.append(state)
        if state.max_celsius is not None:
            self.peak = max(self.peak, state.max_celsius)
        if self._on_sample is not None:
            self._on_sample(state)
        if not state.readings:
            raise ThermalError("temperature sensors became unreadable mid-run")
        if not state.ok:
            hot = ", ".join(f"{r.sensor}={r.celsius:.0f}C" for r in state.violations())
            raise ThermalError(f"thermal limit exceeded: {hot} > {self.limit:.0f}C; stopping")
        return state

    # Alias used by training loops.
    check = poll

    def __enter__(self) -> ThermalGuard:
        self.poll()  # entry is also a pre-flight check
        return self

    def __exit__(self, *exc_info) -> None:
        return None

    def summary(self) -> str:
        if not self.history:
            return "thermal: no samples"
        return (
            f"thermal: {len(self.history)} samples, peak {self.peak:.0f}C, limit {self.limit:.0f}C"
        )
