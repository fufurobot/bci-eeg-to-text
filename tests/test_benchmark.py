"""Red tests for the EEG-to-text benchmark harness.

Baselines that are already downloaded live outside the repo (ECHO's Lightning
checkpoint, Thought2Text's HF-style code). Neither is importable as-is here:
ECHO ships weights without its `FAST`/`EEG2Text` code, and Thought2Text needs
its `channelnet` package plus a base LLM. The harness therefore treats every
model as an **adapter** behind one small interface, so a benchmark row can be
produced whether or not the original code can be installed, and so our own model
is measured on exactly the same footing.

Contract under test:

* an adapter maps ``(batch, channels, time)`` EEG to either per-frame logits
  (CTC models) or generated text (seq2seq models);
* the harness turns references + hypotheses into CER / exact-match, and always
  names which model produced them;
* every model -- ECHO, Thought2Text, ours -- is reachable through the same call,
  so a comparison cannot accidentally use different metrics;
* when a model's code or weights are unavailable the harness reports a clear
  skip rather than a silent zero, because a fake score is worse than none.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

ECHO_CKPT = Path("/home/fufu/ECHO/checkpoints/ECHO.ckpt")
T2T_DIR = Path("/home/fufu/Thought2Text")


class FakeEEG:
    """A model stub: enough surface for the harness to score it."""

    name = "fake"
    kind = "ctc"

    def forward_logits(self, eeg: torch.Tensor) -> torch.Tensor:
        b, _, t = eeg.shape
        # 4 classes to match CharTokenizer.build(["你好"]) (blank, unk, 你, 好).
        return torch.randn(b, max(2, t // 32), 4)

    def input_lengths_to_output(self, lengths: torch.Tensor) -> torch.Tensor:
        return (lengths // 32).clamp(min=1)


class TestAdapterContract:
    def test_ctc_adapter_produces_per_frame_logits(self) -> None:
        from bci_eeg_to_text.benchmark import CTCAdapter

        adapter = CTCAdapter("ours", FakeEEG(), n_frames_stride=32)
        logits = adapter.logits(torch.randn(2, 4, 256))
        assert logits.ndim == 3
        assert logits.shape[0] == 2

    def test_ctc_adapter_decodes(self) -> None:
        from bci_eeg_to_text.benchmark import CTCAdapter
        from bci_eeg_to_text.tokenizer import CharTokenizer

        tok = CharTokenizer.build(["你好", "世界"])
        adapter = CTCAdapter("ours", FakeEEG(), n_frames_stride=32, tokenizer=tok)
        texts = adapter.transcribe(torch.randn(1, 4, 256))
        assert isinstance(texts, list) and len(texts) == 1
        assert isinstance(texts[0], str)

    def test_seq2seq_adapter_requires_a_generate_callable(self) -> None:
        from bci_eeg_to_text.benchmark import Seq2SeqAdapter

        with pytest.raises(TypeError):
            Seq2SeqAdapter("x", object())  # no .generate

    def test_seq2seq_adapter_transcribes(self) -> None:
        from bci_eeg_to_text.benchmark import Seq2SeqAdapter

        class Gen:
            def generate(self, eeg):
                return ["你好"] * eeg.shape[0]

        adapter = Seq2SeqAdapter("echo-like", Gen())
        assert adapter.transcribe(torch.randn(3, 4, 128)) == ["你好", "你好", "你好"]


class TestScoring:
    def test_reports_cer_and_exact_match(self) -> None:
        from bci_eeg_to_text.benchmark import score

        result = score(["你好", "世界"], ["你好", "世舟"])
        assert result["cer"] == pytest.approx(0.25)
        assert result["exact_match"] == pytest.approx(0.5)
        assert result["n"] == 2

    def test_empty_predictions_score_badly_not_silently(self) -> None:
        """An empty hypothesis must count as an error, never as a skip."""
        from bci_eeg_to_text.benchmark import score

        result = score(["你好"], [""])
        assert result["cer"] == pytest.approx(1.0)
        assert result["exact_match"] == 0.0

    def test_length_mismatch_is_rejected(self) -> None:
        from bci_eeg_to_text.benchmark import score

        with pytest.raises(ValueError, match="equal length"):
            score(["a"], ["a", "b"])


class TestRegistry:
    def test_registry_lists_available_models(self) -> None:
        from bci_eeg_to_text.benchmark import available_models

        names = available_models()
        assert "ours" in names
        assert "echo" in names
        assert "thought2text" in names

    def test_unavailable_model_reports_a_reason(self) -> None:
        """A missing weight must surface as a skip with a reason, not a zero."""
        from bci_eeg_to_text.benchmark import probe

        status = probe("echo", echo_ckpt=Path("/nonexistent/ECHO.ckpt"))
        assert status.available is False
        assert status.reason

    def test_unknown_model_is_an_error(self) -> None:
        from bci_eeg_to_text.benchmark import probe

        with pytest.raises(KeyError):
            probe("not-a-model")


@pytest.mark.skipif(not ECHO_CKPT.exists(), reason="ECHO checkpoint not downloaded")
class TestEchoOnDisk:
    def test_checkpoint_is_probed_without_importing_missing_code(self) -> None:
        """ECHO ships weights but not its FAST/EEG2Text code.

        Probing must therefore read the checkpoint without needing those modules
        importable, succeed in describing its shape, and report *unavailable with
        a reason* -- the code to instantiate the network genuinely was not
        downloaded, so claiming it is runnable would be a fabrication.
        """
        from bci_eeg_to_text.benchmark import probe

        status = probe("echo", echo_ckpt=ECHO_CKPT)
        assert status.kind == "seq2seq"
        assert status.available is False
        assert "FAST" in status.reason or "EEG2Text" in status.reason
        assert status.details and status.details["n_tensors"] > 0

    def test_checkpoint_reports_its_shape(self) -> None:
        from bci_eeg_to_text.benchmark import echo_checkpoint_info

        info = echo_checkpoint_info(ECHO_CKPT)
        assert info["decoder_vocab"] > 1000
        assert info["d_model"] > 0
        assert info["n_tensors"] > 0


@pytest.mark.skipif(not T2T_DIR.exists(), reason="Thought2Text not downloaded")
class TestThought2TextOnDisk:
    def test_probe_reports_missing_base_llm_as_reason(self) -> None:
        """Thought2Text needs a base LLM plus LoRA adapters; without them the
        harness must say so rather than pretend the baseline ran."""
        from bci_eeg_to_text.benchmark import probe

        status = probe("thought2text", t2t_dir=T2T_DIR)
        assert status.kind in {"seq2seq", "causal-lm"}
        if not status.available:
            assert status.reason


class TestRunBenchmark:
    def test_runs_a_model_and_labels_the_row(self) -> None:
        from bci_eeg_to_text.benchmark import CTCAdapter, run_benchmark
        from bci_eeg_to_text.tokenizer import CharTokenizer

        adapter = CTCAdapter(
            "ours",
            FakeEEG(),
            n_frames_stride=32,
            tokenizer=CharTokenizer.build(["你好"]),
        )
        rows = run_benchmark(
            [adapter],
            eeg=[torch.randn(1, 4, 256)],
            references=["你好"],
        )
        assert rows[0]["model"] == "ours"
        assert "cer" in rows[0]

    def test_skips_are_recorded_not_dropped(self) -> None:
        from bci_eeg_to_text.benchmark import CTCAdapter, ModelStatus, run_benchmark
        from bci_eeg_to_text.tokenizer import CharTokenizer

        adapter = CTCAdapter(
            "ours",
            FakeEEG(),
            n_frames_stride=32,
            tokenizer=CharTokenizer.build(["你好"]),
        )
        rows = run_benchmark(
            [adapter],
            eeg=[torch.randn(1, 4, 256)],
            references=["你好"],
            skipped=[ModelStatus(name="echo", available=False, reason="no code")],
        )
        assert any(r.get("skipped") for r in rows)

    def test_table_is_human_readable(self) -> None:
        from bci_eeg_to_text.benchmark import format_table

        table = format_table([{"model": "ours", "cer": 0.5, "exact_match": 0.2, "n": 10}])
        assert "model" in table.lower()
        assert "ours" in table
