"""Red tests for multi-backend training: cpu, gpu, and passwordless SSH clusters.

The training mode is selected on the CLI and must be validated **before** any
work starts, because the failure it prevents is expensive: discovering that no
GPU exists after the dataset has been indexed and the model built wastes
minutes, and silently falling back to CPU turns a ten-minute job into hours.

Distributed training must offer two strategies with different tradeoffs:

* ``dataparallel`` -- one process, replicas across GPUs in this process.
  Simple, but the model is copied whole into every GPU, so it cannot exceed one
  device's memory.
* ``fsdp`` -- shards parameters, gradients and optimizer state across ranks, so
  a model larger than one GPU can train. Requires one process per GPU.

Cluster launch is over **passwordless SSH**, and must support **multi-hop**
(topology A -> B -> C) rather than only a flat star, since that is how lab
machines are usually reachable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bci_eeg_to_text import distributed as dist


class TestTrainingModeValidation:
    def test_modes_are_enumerated(self) -> None:
        assert set(dist.TRAINING_MODES) == {"cpu", "gpu", "ssh"}

    def test_cpu_mode_is_accepted_without_a_gpu(self) -> None:
        mode = dist.resolve_mode("cpu", cuda_available=False)
        assert mode.kind == "cpu"
        assert mode.requires_gpu is False

    def test_gpu_mode_fails_loudly_without_a_gpu(self) -> None:
        """Never silently fall back: CPU is ~125x slower here."""
        with pytest.raises(dist.ModeUnavailable, match="CUDA"):
            dist.resolve_mode("gpu", cuda_available=False)

    def test_gpu_mode_reports_devices(self) -> None:
        mode = dist.resolve_mode("gpu", cuda_available=True, n_devices=1)
        assert mode.kind == "gpu"
        assert mode.n_devices == 1

    def test_unknown_mode_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown training mode"):
            dist.resolve_mode("tpu", cuda_available=True)

    def test_ssh_mode_requires_hosts(self) -> None:
        with pytest.raises(ValueError, match="hosts"):
            dist.resolve_mode("ssh", cuda_available=True, hosts=[])


class TestStrategies:
    def test_strategies_are_enumerated(self) -> None:
        assert set(dist.STRATEGIES) == {"dataparallel", "fsdp"}

    def test_dataparallel_is_single_process(self) -> None:
        plan = dist.plan_distributed("dataparallel", n_devices=2)
        assert plan.n_processes == 1
        assert plan.shards_parameters is False

    def test_fsdp_uses_one_process_per_device(self) -> None:
        plan = dist.plan_distributed("fsdp", n_devices=2)
        assert plan.n_processes == 2
        assert plan.shards_parameters is True

    def test_fsdp_requires_at_least_two_devices(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            dist.plan_distributed("fsdp", n_devices=1)

    def test_unknown_strategy_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown strategy"):
            dist.plan_distributed("deepspeed", n_devices=2)


class TestMultiHopSSH:
    def test_parses_a_flat_host_list(self) -> None:
        hops = dist.parse_topology("a:1,b:2")
        assert hops == ["a:1", "b:2"]

    def test_parses_a_multi_hop_chain(self) -> None:
        """a -> b -> c: the last hop is the worker, the rest are jump hosts."""
        chain = dist.parse_topology("jump@a:22|worker@c:3")
        assert chain == ["jump@a:22", "worker@c:3"]

    def test_builds_a_proxy_jump_through_each_hop(self) -> None:
        chain = dist.parse_topology("j1:22|j2:22|w:8")
        argv = dist.build_ssh_command(chain, remote_cmd="echo hi", identity=Path("/k"))
        joined = " ".join(argv)
        assert "-J" in argv
        # Two jumps for a three-hop chain.
        jump = argv[argv.index("-J") + 1]
        assert jump.count(",") == 1

    def test_single_hop_needs_no_proxy_jump(self) -> None:
        argv = dist.build_ssh_command(["w:4"], remote_cmd="echo hi", identity=Path("/k"))
        assert "-J" not in argv

    def test_command_propagates_strict_options_to_every_hop(self) -> None:
        """Jump hosts must not prompt; BatchMode keeps a hung hop from blocking."""
        argv = dist.build_ssh_command(
            ["j:1", "w:2"], remote_cmd="true", identity=Path("/k")
        )
        assert "BatchMode=yes" in " ".join(argv)

    def test_worker_count_comes_from_the_last_hop(self) -> None:
        chain = dist.parse_topology("j:1|w:8")
        assert dist.worker_count(chain) == 8

    def test_rejects_an_empty_topology(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            dist.parse_topology("")

    def test_rejects_a_zero_worker_count(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            dist.parse_topology("w:0")


class TestLaunchCommand:
    def test_local_launch_uses_torchrun_for_fsdp(self) -> None:
        argv = dist.build_launch_command(
            plan=dist.plan_distributed("fsdp", n_devices=2),
            script=Path("train.py"),
            extra=["--lr", "1e-3"],
        )
        assert "torchrun" in argv[0]
        assert "--nproc_per_node=2" in argv

    def test_local_launch_keeps_dataparallel_in_one_process(self) -> None:
        argv = dist.build_launch_command(
            plan=dist.plan_distributed("dataparallel", n_devices=2),
            script=Path("train.py"),
            extra=[],
        )
        assert "torchrun" not in argv[0]
        assert argv[-1] == "train.py"

    def test_ssh_launch_wraps_the_command_in_ssh(self) -> None:
        argv = dist.build_ssh_launch(
            chain=["w:2"],
            plan=dist.plan_distributed("fsdp", n_devices=2),
            script=Path("train.py"),
            extra=[],
            identity=Path("/k"),
        )
        assert argv[0] == "ssh"
        assert "torchrun" in " ".join(argv)


class TestClusterProbe:
    def test_probe_reports_per_host_status(self, tmp_path: Path) -> None:
        report = dist.probe_cluster(
            ["w:1"],
            runner=lambda host: (_ for _ in ()).throw(OSError("unreachable")),
        )
        assert report[0].reachable is False
        assert report[0].reason

    def test_probe_marks_reachable_hosts(self) -> None:
        report = dist.probe_cluster(["w:1"], runner=lambda host: True)
        assert report[0].reachable is True

    def test_unreachable_host_stops_launch(self) -> None:
        with pytest.raises(dist.ClusterUnreachable, match="w:1"):
            dist.require_reachable(
                [dist.HostStatus("w:1", reachable=False, reason="no route")]
            )
