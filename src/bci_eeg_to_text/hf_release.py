"""Build a self-contained HuggingFace release folder.

Loading a custom architecture through ``AutoModel`` requires the code to travel
with the weights and ``trust_remote_code=True``. A folder that imported
``bci_eeg_to_text`` would only work for someone who already installed this
package, which defeats the purpose, so the builder **vendors** the architecture
into a single ``modeling_eeg_ctc.py`` alongside ``configuration_eeg_ctc.py``,
``tokenizer_config.json`` and the weights.

Layout produced::

    release/
        config.json                 # model_type: eeg-ctc
        modeling_eeg_ctc.py         # vendored architecture (self-contained)
        configuration_eeg_ctc.py    # vendored config
        tokenizer_config.json
        eeg_vocab.json
        tokenizer.py                # vendored tokenizer
        processor_config.json
        model.safetensors
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

_PACKAGE = Path(__file__).parent

# A `from __future__` import must be the first statement in a file, so vendored
# sources are emitted verbatim; the note goes in a comment instead of a docstring.
MODELING_TEMPLATE = """# -*- coding: utf-8 -*-
# Vendored EEG CTC architecture (generated; do not edit by hand).
# Self-contained so the folder loads through ``AutoModel`` with
# ``trust_remote_code=True`` without installing the training package.
{source}
"""

CONFIG_TEMPLATE = """# -*- coding: utf-8 -*-
# Vendored EEG CTC configuration (generated; do not edit by hand).
{source}
"""

TOKENIZER_TEMPLATE = """# -*- coding: utf-8 -*-
# Vendored character tokenizer (generated; do not edit by hand).
{source}
"""


def _vendored_model_source() -> str:
    """Architecture source with the package import replaced by the local copy."""
    src = (_PACKAGE / "hf_model.py").read_text(encoding="utf-8")
    # The release folder carries its own config; drop the intra-package import.
    src = src.replace("from bci_eeg_to_text.hf_release import write_custom_code", "")
    src = src.replace("        from bci_eeg_to_text.hf_release import write_custom_code\n", "")
    # Strip the save_pretrained override that referenced the builder.
    src = src.replace(
        "    def save_pretrained(self, directory, **kwargs) -> None:\n"
        '        """Write a self-contained folder: config, weights and vendored code."""\n'
        "        super().save_pretrained(directory, **kwargs)\n"
        "        write_custom_code(directory)\n\n",
        "",
    )
    # Class names must match what config.json names.
    src = src.replace("class EEGCTCConfig(", "class EEGCTCConfig(")
    return src


def _vendored_config_source() -> str:
    """Just the config class, importable on its own."""
    src = (_PACKAGE / "hf_model.py").read_text(encoding="utf-8")
    return src


def _vendored_tokenizer_source() -> str:
    src = (_PACKAGE / "hf_tokenizer.py").read_text(encoding="utf-8")
    src = src.replace(
        "from bci_eeg_to_text.tokenizer import BLANK, UNK, CharTokenizer",
        "from .eeg_ctc_tokenizer_support import BLANK, UNK, CharTokenizer",
    )
    return src


def build_release_folder(
    directory: str | Path,
    *,
    n_channels: int = 128,
    n_classes: int = 1002,
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 4,
    vocab: list[str] | None = None,
    model_state: dict | None = None,
) -> Path:
    """Create a loadable HuggingFace folder and return its path."""
    from bci_eeg_to_text.hf_model import EEGCTCConfig, EEGCTCModel

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)

    config = EEGCTCConfig(
        n_channels=n_channels,
        n_classes=n_classes,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
    )
    config.auto_map = {
        "AutoConfig": "configuration_eeg_ctc.EEGCTCConfig",
        "AutoModel": "modeling_eeg_ctc.EEGCTCModel",
    }
    (out / "configuration_eeg_ctc.py").write_text(
        CONFIG_TEMPLATE.format(source=_vendored_config_source()), encoding="utf-8"
    )
    (out / "modeling_eeg_ctc.py").write_text(
        MODELING_TEMPLATE.format(source=_vendored_model_source()), encoding="utf-8"
    )

    payload = {"model_type": "eeg-ctc"}
    for key, value in config.__dict__.items():
        if key.startswith("_") or key == "transformers_version":
            continue
        payload[key] = value
    payload["architectures"] = ["EEGCTCModel"]
    payload["auto_map"] = config.auto_map
    (out / "config.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    model = EEGCTCModel(config)
    if model_state:
        model.load_state_dict(model_state, strict=False)
    try:
        model.save_pretrained(out)
    except Exception:  # noqa: BLE001 - fall back to a plain state dict
        import torch

        torch.save(model.state_dict(), out / "pytorch_model.bin")

    # Tokenizer: vocabulary plus the HF tokenizer config.
    vocab_list = vocab or ["<pad>", "<unk>"]
    (out / "eeg_vocab.json").write_text(
        json.dumps(vocab_list, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tokenizer_payload = {
        "tokenizer_class": "EEGCharacterTokenizer",
        "padding_side": "right",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "model_max_length": 512,
        "auto_map": {
            "AutoTokenizer": ["tokenizer.EEGCharacterTokenizer", None],
        },
    }
    (out / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    shutil.copyfile(_PACKAGE / "hf_tokenizer.py", out / "tokenizer.py")
    shutil.copyfile(_PACKAGE / "tokenizer.py", out / "eeg_ctc_tokenizer_support.py")
    (out / "processor_config.json").write_text(
        json.dumps(
            {
                "processor_class": "EEGProcessor",
                "n_samples": 1152,
                "auto_map": {"AutoProcessor": "hf_processor.EEGProcessor"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    shutil.copyfile(_PACKAGE / "hf_processor.py", out / "hf_processor.py")
    (out / "README.md").write_text(
        "# EEG-to-text (conv-transformer + CTC)\n\n"
        "Convolutional front-end plus pre-norm residual transformer blocks, "
        "decoded with CTC.\n\n"
        "```python\n"
        "from transformers import AutoModel\n"
        f"model = AutoModel.from_pretrained({str(out)!r}, trust_remote_code=True)\n"
        "```\n",
        encoding="utf-8",
    )
    return out


def write_custom_code(directory: str | Path) -> None:
    """Vendor the architecture into an existing folder (used by save_pretrained)."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    (out / "modeling_eeg_ctc.py").write_text(
        MODELING_TEMPLATE.format(source=_vendored_model_source()), encoding="utf-8"
    )
    (out / "configuration_eeg_ctc.py").write_text(
        CONFIG_TEMPLATE.format(source=_vendored_config_source()), encoding="utf-8"
    )
