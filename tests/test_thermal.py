"""Tests for thermal monitoring on a machine with a broken fan.

Cooling is external and manual, so temperature is a safety constraint: a run
that pushes the CPU past its limit can damage hardware with nothing to stop it.
The rules under test:

* read real sensors rather than estimating;
* treat an unreadable machine as **unknown**, never as cool;
* stop work on breach instead of warning and continuing.
"""

from __future__ import annotations

import pytest

from bci_eeg_to_text import thermal


def fake_state(*temps: tuple[str, float], limit: float = 80.0) -> thermal.ThermalState:
    """Build a state; ``gpu*`` sensors get the hard limit, others the CPU one."""
    return thermal.ThermalState(
        readings=[
            thermal.Reading(
                sensor=s, celsius=c, kind="gpu" if s.startswith("gpu") else "cpu"
            )
            for s, c in temps
        ],
        limit=limit,
    )


class TestState:
    def test_reports_the_hottest_sensor(self) -> None:
        state = fake_state(("cpu", 60), ("gpu", 75))
        assert state.max_celsius == 75
        assert state.hottest.sensor == "gpu"

    def test_ok_when_every_sensor_is_within_the_limit(self) -> None:
        assert fake_state(("cpu", 60), ("gpu", 79.9)).ok is True

    def test_gpu_over_80_is_a_violation(self) -> None:
        """The GPU ceiling is the hard one at 80 C."""
        state = fake_state(("cpu", 50), ("gpu", 81))
        assert state.ok is False
        assert [r.sensor for r in state.violations()] == ["gpu"]

    def test_cpu_gets_the_looser_ceiling(self) -> None:
        """This machine's cooling cannot hold the CPU under 80, so its limit is
        higher by policy; the GPU limit stays hard."""
        state = fake_state(("cpu", 85))
        assert state.ok is True
        assert fake_state(("cpu", 89)).ok is False

    def test_exactly_at_the_limit_is_acceptable(self) -> None:
        assert fake_state(("gpu", 80.0)).ok is True

    def test_no_sensors_is_unknown_not_cool(self) -> None:
        """An unreadable machine must never be reported as safe."""
        state = thermal.ThermalState(readings=[])
        assert state.ok is False
        assert "unknown" in state.describe()

    def test_describe_names_each_sensor(self) -> None:
        text = fake_state(("cpu", 72), ("gpu", 61)).describe()
        assert "cpu=72C" in text
        assert "gpu=61C" in text


class TestReadingSensors:
    def test_reads_linux_thermal_zones(self, tmp_path) -> None:
        zone = tmp_path / "thermal_zone0"
        zone.mkdir()
        (zone / "type").write_text("x86_pkg_temp\n")
        (zone / "temp").write_text("73000\n")  # millidegrees
        readings = thermal.read_cpu_temps(tmp_path)
        assert readings[0].sensor == "x86_pkg_temp"
        assert readings[0].celsius == pytest.approx(73.0)

    def test_skips_unreadable_zones(self, tmp_path) -> None:
        zone = tmp_path / "thermal_zone0"
        zone.mkdir()
        (zone / "temp").write_text("not-a-number\n")
        assert thermal.read_cpu_temps(tmp_path) == []

    def test_missing_thermal_root_yields_nothing(self, tmp_path) -> None:
        assert thermal.read_cpu_temps(tmp_path / "absent") == []

    def test_gpu_reading_tolerates_a_missing_nvidia_smi(self, monkeypatch) -> None:
        def boom(*args, **kwargs):
            raise FileNotFoundError("nvidia-smi")

        monkeypatch.setattr(thermal.subprocess, "run", boom)
        assert thermal.read_gpu_temps() == []


class TestRequireCool:
    def test_raises_when_already_hot(self, monkeypatch) -> None:
        monkeypatch.setattr(
            thermal, "snapshot",
            lambda limit=80.0, cpu_limit=88.0: fake_state(("gpu", 85)),
        )
        with pytest.raises(thermal.ThermalError, match="exceeds"):
            thermal.require_cool()

    def test_passes_when_cool(self, monkeypatch) -> None:
        monkeypatch.setattr(
            thermal, "snapshot",
            lambda limit=80.0, cpu_limit=88.0: fake_state(("gpu", 60)),
        )
        assert thermal.require_cool().ok

    def test_refuses_to_run_blind_when_no_sensor_is_readable(self, monkeypatch) -> None:
        monkeypatch.setattr(
            thermal, "snapshot", lambda limit=80.0, cpu_limit=88.0: thermal.ThermalState()
        )
        with pytest.raises(thermal.ThermalError, match="no temperature sensors"):
            thermal.require_cool()


class TestGuard:
    def test_poll_records_history_and_peak(self) -> None:
        temps = iter([60.0, 70.0, 65.0])

        def sample():
            return fake_state(("cpu", next(temps)))

        guard = thermal.ThermalGuard(sample=sample)
        for _ in range(3):
            guard.poll()
        assert len(guard.history) == 3
        assert guard.peak == 70.0

    def test_raises_immediately_on_breach(self) -> None:
        guard = thermal.ThermalGuard(sample=lambda: fake_state(("gpu", 85)))
        with pytest.raises(thermal.ThermalError, match="thermal limit exceeded"):
            guard.poll()

    def test_stops_at_the_first_breach_without_continuing(self) -> None:
        temps = iter([60.0, 81.0, 60.0])  # breach is on the GPU

        def sample():
            return fake_state(("gpu", next(temps)))

        guard = thermal.ThermalGuard(sample=sample)
        guard.poll()
        with pytest.raises(thermal.ThermalError):
            guard.poll()
        # The third sample was never taken: the guard stopped.
        assert len(guard.history) == 2

    def test_check_is_an_alias_for_poll(self) -> None:
        guard = thermal.ThermalGuard(sample=lambda: fake_state(("cpu", 60)))
        guard.check()
        assert len(guard.history) == 1

    def test_entering_the_context_performs_a_preflight_check(self) -> None:
        guard = thermal.ThermalGuard(sample=lambda: fake_state(("gpu", 85)))
        with pytest.raises(thermal.ThermalError), guard:
            pass

    def test_mid_run_sensor_loss_raises(self) -> None:
        guard = thermal.ThermalGuard(sample=lambda: thermal.ThermalState())
        with pytest.raises(thermal.ThermalError, match="unreadable"):
            guard.poll()

    def test_on_sample_callback_sees_each_reading(self) -> None:
        seen = []
        guard = thermal.ThermalGuard(sample=lambda: fake_state(("cpu", 60)), on_sample=seen.append)
        guard.poll()
        assert len(seen) == 1

    def test_summary_reports_peak(self) -> None:
        guard = thermal.ThermalGuard(sample=lambda: fake_state(("cpu", 66)))
        guard.poll()
        summary = guard.summary()
        assert "peak 66C" in summary
        # Both ceilings are reported, since they differ by design.
        assert "80C" in summary


class TestRealSensors:
    """These run against the actual machine, without enforuing a limit."""

    def test_cpu_sensors_are_readable_here(self) -> None:
        readings = thermal.read_cpu_temps()
        if not readings:
            pytest.skip("no thermal zones exposed in this environment")
        assert all(r.celsius > 0 for r in readings)

    def test_snapshot_includes_a_timestamp(self) -> None:
        state = thermal.snapshot()
        if not state.readings:
            pytest.skip("no sensors readable")
        assert state.timestamp > 0
