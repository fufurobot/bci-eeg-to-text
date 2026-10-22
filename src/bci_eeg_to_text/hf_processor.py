"""HuggingFace ``Processor`` for EEG trials.

A processor is the glue between raw inputs and a model's tensors. For EEG the
"pixel values" are the multichannel signal itself:

* crop or zero-pad a trial to a fixed number of samples, so trials batch,
* optionally wrap the paired character tokenizer, so one call yields EEG plus
  CTC labels from ``(eeg, text)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

try:
    from transformers import ProcessorMixin

    _HF = True
except ImportError:  # pragma: no cover
    ProcessorMixin = object  # type: ignore[assignment,misc]
    _HF = False


class EEGProcessor(ProcessorMixin):
    """Crop/pad EEG to a fixed length and optionally tokenize the target text."""

    attributes = ["tokenizer"]
    tokenizer_class = "PreTrainedTokenizer"

    def __init__(self, n_samples: int = 1152, tokenizer=None, **kwargs) -> None:
        self.n_samples = int(n_samples)
        if tokenizer is not None:
            super().__init__(tokenizer)
        else:
            # ProcessorMixin insists on its attributes being present; a bare
            # processor (used when only EEG normalisation is wanted) keeps them
            # as None rather than raising.
            self.tokenizer = None  # type: ignore[assignment]

    def _fit(self, eeg) -> torch.Tensor:
        """Crop or zero-pad ``(channels, samples)`` to ``n_samples``."""
        arr = eeg.detach().cpu().numpy() if torch.is_tensor(eeg) else np.asarray(eeg)
        arr = np.atleast_2d(arr).astype(np.float32)
        n_ch, length = arr.shape
        out = np.zeros((n_ch, self.n_samples), dtype=np.float32)
        keep = min(length, self.n_samples)
        out[:, :keep] = arr[:, :keep]
        return torch.from_numpy(out)

    def __call__(self, eeg=None, text=None, **kwargs):
        if eeg is None:
            raise ValueError("eeg is required")
        result: dict[str, object] = {"pixel_values": self._fit(eeg)}
        if text is not None:
            if self.tokenizer is None:
                raise ValueError("text was given but the processor has no tokenizer")
            encoded = self.tokenizer(text)
            ids = encoded["input_ids"]
            result["labels"] = torch.tensor(ids, dtype=torch.long)
            result["attention_mask"] = torch.tensor(encoded["attention_mask"], dtype=torch.long)
        return result

    def save_pretrained(self, directory, **kwargs) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        import json

        (path / "processor_config.json").write_text(
            json.dumps(
                {
                    "processor_class": type(self).__name__,
                    "n_samples": int(self.n_samples),
                },
                indent=2,
            )
        )
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(directory)

    @classmethod
    def from_pretrained(cls, directory, **kwargs):
        import json

        path = Path(directory)
        cfg = json.loads((path / "processor_config.json").read_text())
        tokenizer = kwargs.pop("tokenizer", None)
        if tokenizer is None:
            try:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            except Exception:  # noqa: BLE001 - a tokenizer is optional
                tokenizer = None
        return cls(n_samples=int(cfg.get("n_samples", 1152)), tokenizer=tokenizer)
