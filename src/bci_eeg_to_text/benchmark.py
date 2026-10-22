"""Benchmark harness: score EEG-to-text models on identical footing.

Every model -- a downloaded baseline or ours -- is wrapped in an adapter that
exposes the same two methods, so a comparison cannot accidentally mix metrics:

* :class:`CTCAdapter` for models that emit per-frame logits (ours),
* :class:`Seq2SeqAdapter` for models that generate text (ECHO, Thought2Text).

Scoring always reports character error rate and exact-match accuracy, which are
the only numbers that mean anything for text output. An empty hypothesis counts
as an error; an unavailable model is reported as a skip **with a reason** rather
than a zero, because a fabricated score is worse than an honest gap.

Baselines are not vendored. ECHO ships a PyTorch Lightning checkpoint without its
`FAST`/`EEG2Text` code, and Thought2Text needs its `channelnet` package plus a
base LLM, so the harness probes what is actually present and says so.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch

from bci_eeg_to_text.decode import character_error_rate, sentence_accuracy
from bci_eeg_to_text.tokenizer import CharTokenizer


class AdapterError(RuntimeError):
    """Raised when a model cannot produce predictions."""


@runtime_checkable
class Adapter(Protocol):
    """What the harness needs from any model."""

    name: str
    kind: str

    def transcribe(self, eeg: torch.Tensor) -> list[str]:
        """``(B, C, T)`` EEG -> one decoded string per sample."""


@dataclass(frozen=True)
class ModelStatus:
    """Whether a model can be benchmarked here, and why not when it cannot."""

    name: str
    available: bool
    reason: str = ""
    kind: str = "unknown"
    n_parameters: int = 0
    details: dict[str, object] | None = None


class CTCAdapter:
    """Wraps a per-frame model: logits -> greedy CTC decode."""

    kind = "ctc"

    def __init__(
        self,
        name: str,
        model: object,
        *,
        n_frames_stride: int = 32,
        tokenizer: CharTokenizer | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.stride = n_frames_stride
        self.tokenizer = tokenizer

    def logits(self, eeg: torch.Tensor) -> torch.Tensor:
        """Per-frame logits, whatever the underlying model calls itself."""
        model = self.model
        if hasattr(model, "forward_logits"):
            return model.forward_logits(eeg)  # type: ignore[no-any-return]
        out = model(eeg)  # type: ignore[operator]
        return out.logits if hasattr(out, "logits") else out

    def frame_lengths(self, n_samples: torch.Tensor) -> torch.Tensor:
        model = self.model
        if hasattr(model, "input_lengths_to_output"):
            return model.input_lengths_to_output(n_samples)  # type: ignore[no-any-return]
        return torch.div(n_samples, self.stride, rounding_mode="floor").clamp(min=1)

    def transcribe(self, eeg: torch.Tensor) -> list[str]:
        if self.tokenizer is None:
            raise AdapterError(f"{self.name}: a tokenizer is required to decode CTC output")
        from bci_eeg_to_text.decode import greedy_decode

        with torch.no_grad():
            logits = self.logits(eeg)
            lengths = self.frame_lengths(torch.tensor([eeg.shape[-1]] * eeg.shape[0]))
        return greedy_decode(logits, self.tokenizer, lengths)


class Seq2SeqAdapter:
    """Wraps a generating model: EEG -> text in one call."""

    kind = "seq2seq"

    def __init__(self, name: str, model: object) -> None:
        if not hasattr(model, "generate"):
            raise TypeError(
                f"{name}: a seq2seq adapter needs a model with .generate(eeg) -> list[str]"
            )
        self.name = name
        self.model = model

    def transcribe(self, eeg: torch.Tensor) -> list[str]:
        with torch.no_grad():
            out = self.model.generate(eeg)  # type: ignore[union-attr]
        if isinstance(out, str):
            return [out]
        return [str(x) for x in out]


def score(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    """CER and exact-match accuracy. Empty output is an error, not a gap."""
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have equal length")
    if not references:
        return {"cer": float("nan"), "exact_match": 0.0, "n": 0.0}
    return {
        "cer": float(character_error_rate(references, hypotheses)),
        "exact_match": float(sentence_accuracy(references, hypotheses)),
        "n": float(len(references)),
    }


def echo_checkpoint_info(path: Path) -> dict[str, object]:
    """Read an ECHO Lightning checkpoint's shape without importing its code.

    The distributed checkpoint pickles references to ``dataset``, ``fast`` and
    ``EEG2Text`` modules that were never published alongside the weights, so a
    plain ``torch.load`` fails. The tensors are stored as numbered blobs in the
    zip archive, and the decoder vocabulary and width are recoverable from the
    pickle bytes and the archive sizes, which is enough to report what the model
    is and confirm the file is intact.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        blobs = [n for n in names if n.startswith("archive/data/")]
        pickle_name = next((n for n in names if n.endswith("data.pkl")), None)
        raw = archive.read(pickle_name).decode("latin-1") if pickle_name else ""

    # The decoder embedding shape appears in the pickle as a tensor descriptor;
    # infer the vocabulary from the largest embedding-sized blob instead.
    sizes = []
    with zipfile.ZipFile(path) as archive:
        for n in blobs:
            info = archive.getinfo(n)
            sizes.append(info.file_size)
    # embedding (51865, 384) of float32 is by far the largest single tensor.
    vocab_estimate = max(sizes) // (4 * 384) if sizes else 0
    return {
        "n_tensors": len(blobs),
        "decoder_vocab": int(vocab_estimate),
        "d_model": 384 if "conv1.weight" in raw or "token_embedding" in raw else 0,
        "has_whisper_decoder": "decoder" in raw,
        "size_bytes": path.stat().st_size,
    }


def probe(
    name: str,
    *,
    echo_ckpt: Path | None = None,
    t2t_dir: Path | None = None,
) -> ModelStatus:
    """Report whether a model can be benchmarked, with a reason when it cannot."""
    if name == "echo":
        ckpt = echo_ckpt or Path("/home/fufu/ECHO/checkpoints/ECHO.ckpt")
        if not ckpt.exists():
            return ModelStatus(
                name, False, reason=f"checkpoint not found at {ckpt}", kind="seq2seq"
            )
        try:
            info = echo_checkpoint_info(ckpt)
        except Exception as exc:  # noqa: BLE001
            return ModelStatus(name, False, reason=f"unreadable checkpoint: {exc}", kind="seq2seq")
        code_present = (ckpt.parent.parent / "FAST").exists() or (
            ckpt.parent.parent / "EEG2Text"
        ).exists()
        if not code_present:
            return ModelStatus(
                name,
                False,
                reason=(
                    "weights present but the FAST/EEG2Text implementation was not "
                    "downloaded, so the architecture cannot be instantiated"
                ),
                kind="seq2seq",
                details=info,
            )
        return ModelStatus(name, True, kind="seq2seq", details=info)

    if name == "thought2text":
        root = t2t_dir or Path("/home/fufu/Thought2Text")
        if not root.exists():
            return ModelStatus(name, False, reason=f"not found at {root}", kind="causal-lm")
        needs = ["channelnet", "config.py"]
        missing = [m for m in needs if not (root / m).exists()]
        if missing:
            return ModelStatus(
                name,
                False,
                reason=f"missing modules: {', '.join(missing)}",
                kind="causal-lm",
            )
        return ModelStatus(
            name,
            False,
            reason="code present but needs a base LLM plus trained LoRA adapters",
            kind="causal-lm",
        )

    if name == "ours":
        return ModelStatus(name, True, kind="ctc")

    raise KeyError(f"unknown model {name!r}")


def available_models() -> list[str]:
    """Names the harness knows about."""
    return ["ours", "echo", "thought2text"]


def run_benchmark(
    adapters: list[Adapter],
    *,
    eeg: list[torch.Tensor],
    references: list[str],
    skipped: list[ModelStatus] | None = None,
) -> list[dict[str, object]]:
    """Score each adapter on the same inputs, labelling every row by model."""
    rows: list[dict[str, object]] = []
    for adapter in adapters:
        hypotheses: list[str] = []
        for sample in eeg:
            batch = sample if sample.ndim == 3 else sample.unsqueeze(0)
            hypotheses.extend(adapter.transcribe(batch))
        result = score(references, hypotheses)
        rows.append({"model": adapter.name, "kind": adapter.kind, **result})
    for status in skipped or []:
        rows.append(
            {
                "model": status.name,
                "kind": status.kind,
                "skipped": True,
                "reason": status.reason,
            }
        )
    return rows


def format_table(rows: list[dict[str, object]]) -> str:
    """Render benchmark rows as a fixed-width table."""
    header = f"{'model':<18}{'cer':>8}{'exact':>8}{'n':>6}  note"
    lines = [header, "-" * len(header)]
    for row in rows:
        # Treat a row as skipped when it carries no metrics, not only when a
        # flag is set: callers building rows by hand omit the flag.
        if row.get("skipped") or "cer" not in row:
            reason = str(row.get("reason", ""))[:60]
            lines.append(f"{str(row['model']):<18}{'-':>8}{'-':>8}{'-':>6}  skipped: {reason}")
            continue
        lines.append(
            f"{str(row['model']):<18}{row['cer']:>8.3f}{row['exact_match']:>8.3f}{int(row['n']):>6}"
        )
    return "\n".join(lines)
