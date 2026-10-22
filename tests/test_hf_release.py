"""Tests for the HuggingFace-native conv-transformer release format.

The goal is that a trained checkpoint can be published as an ordinary HuggingFace
folder and loaded with the stock `transformers` entry points:

    AutoConfig.from_pretrained(folder)
    AutoTokenizer.from_pretrained(folder)
    AutoProcessor.from_pretrained(folder)
    AutoModel.from_pretrained(folder)

with `trust_remote_code=True` where the custom architecture is needed. These
tests are written before the implementation; they define the contract:

* the folder contains a config whose ``model_type`` resolves through AutoConfig,
* the tokenizer and processor round-trip Chinese text and CTC targets,
* ``AutoModel`` returns per-frame logits for raw EEG,
* custom code is vendored so the folder is self-contained (no import of
  ``bci_eeg_to_text`` is required to load it).

The heavy end-to-end AutoModel checks are marked ``hf`` so a machine without the
custom-code auto-map can still run the rest.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _have_transformers() -> bool:
    try:
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


needs_transformers = pytest.mark.skipif(
    not _have_transformers(), reason="transformers not installed"
)


class TestConfigRoundTrip:
    """The config must round-trip and declare itself for auto-loading."""

    def test_config_is_saved_with_model_type(self, tmp_path: Path) -> None:
        from bci_eeg_to_text.hf_model import EEGCTCConfig

        cfg = EEGCTCConfig(
            n_channels=128, n_classes=1002, d_model=64, n_heads=4, n_layers=2
        )
        cfg.save_pretrained(tmp_path)
        payload = json.loads((tmp_path / "config.json").read_text())
        assert payload["model_type"] == "eeg-ctc"
        assert payload["n_channels"] == 128
        assert payload["n_classes"] == 1002

    def test_config_round_trips_through_from_pretrained(self, tmp_path: Path) -> None:
        from bci_eeg_to_text.hf_model import EEGCTCConfig

        cfg = EEGCTCConfig(
            n_channels=8, n_classes=17, d_model=32, n_heads=4, n_layers=1,
            conv_channels=(8, 16), conv_strides=(2, 4),
        )
        cfg.save_pretrained(tmp_path)
        loaded = EEGCTCConfig.from_pretrained(tmp_path)
        assert loaded.n_channels == 8
        assert tuple(loaded.conv_strides) == (2, 4)
        assert loaded.d_model == 32

    def test_config_derives_output_stride(self) -> None:
        from bci_eeg_to_text.hf_model import EEGCTCConfig

        cfg = EEGCTCConfig(conv_strides=(2, 2, 8))
        assert cfg.output_stride == 32


class TestModelContract:
    def test_forward_returns_per_frame_logits(self) -> None:
        from bci_eeg_to_text.hf_model import EEGCTCModel, EEGCTCConfig

        model = EEGCTCModel(
            EEGCTCConfig(
                n_channels=8, n_classes=17, d_model=32, n_heads=4, n_layers=1,
                conv_channels=(8, 16), conv_strides=(2, 4),
            )
        ).eval()
        out = model(pixel_values=torch.randn(2, 8, 1000))
        logits = out.logits if hasattr(out, "logits") else out[0]
        assert logits.ndim == 3
        assert logits.shape[0] == 2
        assert logits.shape[2] == 17

    def test_lengths_are_reported_for_ctc(self) -> None:
        from bci_eeg_to_text.hf_model import EEGCTCModel, EEGCTCConfig

        model = EEGCTCModel(
            EEGCTCConfig(
                n_channels=8, n_classes=17, d_model=32, n_heads=4, n_layers=1,
                conv_channels=(8,), conv_strides=(4,),
            )
        ).eval()
        n = 1000
        with torch.no_grad():
            logits = model(pixel_values=torch.randn(1, 8, n)).logits
        reported = int(model.input_lengths_to_output(torch.tensor([n]))[0])
        assert reported == logits.shape[1]

    def test_save_pretrained_creates_a_loadable_folder(self, tmp_path: Path) -> None:
        from bci_eeg_to_text.hf_model import EEGCTCModel, EEGCTCConfig

        model = EEGCTCModel(
            EEGCTCConfig(n_channels=8, n_classes=17, d_model=32, n_heads=4, n_layers=1)
        )
        model.save_pretrained(tmp_path)
        assert (tmp_path / "config.json").exists()
        assert any(tmp_path.glob("*.safetensors")) or any(tmp_path.glob("*.bin"))


class TestTokenizer:
    def test_round_trips_chinese_text(self) -> None:
        from bci_eeg_to_text.hf_tokenizer import EEGCharTokenizer

        tok = EEGCharTokenizer.build(["我六岁那年", "看见过一幅插图"])
        ids = tok("我六岁那年").input_ids
        assert tok.decode(ids, skip_special_tokens=True) == "我六岁那年"

    def test_exposes_the_ctc_blank(self) -> None:
        from bci_eeg_to_text.hf_tokenizer import EEGCharTokenizer

        tok = EEGCharTokenizer.build(["你好"])
        assert tok.blank_token_id == tok.pad_token_id
        assert tok.blank_token_id is not None

    def test_save_and_load(self, tmp_path: Path) -> None:
        from bci_eeg_to_text.hf_tokenizer import EEGCharTokenizer

        tok = EEGCharTokenizer.build(["你好世界"])
        tok.save_pretrained(tmp_path)
        loaded = EEGCharTokenizer.from_pretrained(tmp_path)
        assert loaded.get_vocab() == tok.get_vocab()
        assert loaded.decode(loaded("你好").input_ids, skip_special_tokens=True) == "你好"

    def test_custom_class_is_auto_registerable(self) -> None:
        from bci_eeg_to_text.hf_tokenizer import EEGCharTokenizer

        assert hasattr(EEGCharTokenizer, "register_for_auto_class")
        assert getattr(EEGCharTokenizer, "_auto_class", None) is not None


class TestProcessor:
    def test_processor_normalises_eeg(self) -> None:
        from bci_eeg_to_text.hf_processor import EEGProcessor

        proc = EEGProcessor(n_samples=1000)
        out = proc(torch.randn(8, 1152))
        assert out["pixel_values"].shape[-1] == 1000
        assert out["pixel_values"].ndim == 2

    def test_processor_and_tokenizer_combine_text_and_eeg(self) -> None:
        from bci_eeg_to_text.hf_processor import EEGProcessor
        from bci_eeg_to_text.hf_tokenizer import EEGCharTokenizer

        tok = EEGCharTokenizer.build(["你好"])
        proc = EEGProcessor(n_samples=512, tokenizer=tok)
        out = proc(eeg=torch.randn(8, 600), text="你好")
        assert out["pixel_values"].shape == (8, 512)
        assert "labels" in out
        assert out["labels"].shape == (2,)

    def test_processor_round_trips(self, tmp_path: Path) -> None:
        from bci_eeg_to_text.hf_processor import EEGProcessor

        proc = EEGProcessor(n_samples=256)
        proc.save_pretrained(tmp_path)
        loaded = EEGProcessor.from_pretrained(tmp_path)
        assert loaded.n_samples == 256


@needs_transformers
class TestAutoClasses:
    """The published folder must work through the stock auto entry points."""

    @pytest.fixture
    def release_dir(self, tmp_path: Path) -> Path:
        from bci_eeg_to_text.hf_release import build_release_folder

        return build_release_folder(
            tmp_path / "release",
            n_channels=8,
            n_classes=17,
            d_model=32,
            n_heads=4,
            n_layers=1,
            vocab=["<pad>", "<unk>", "你", "好"],
        )

    def test_release_folder_is_self_contained(self, release_dir: Path) -> None:
        """Custom code must be vendored so the folder needs no installed package."""
        assert (release_dir / "config.json").exists()
        assert (release_dir / "tokenizer_config.json").exists()
        code = list(release_dir.glob("*.py"))
        assert code, "expected vendored custom code"
        text = "\n".join(p.read_text() for p in code)
        assert "class EEGCTCModel" in text

    def test_auto_config_loads(self, release_dir: Path) -> None:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(release_dir, trust_remote_code=True)
        assert cfg.model_type == "eeg-ctc"

    def test_auto_tokenizer_loads(self, release_dir: Path) -> None:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(release_dir, trust_remote_code=True)
        assert tok.decode(tok("你好").input_ids, skip_special_tokens=True) == "你好"

    def test_auto_processor_loads(self, release_dir: Path) -> None:
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(release_dir, trust_remote_code=True)
        out = proc(eeg=torch.randn(8, 600), text="你好")
        assert "pixel_values" in out


class TestAnyHuggingFaceFolder:
    """The training entry point must accept any folder AutoModel can load."""

    def test_local_folder_is_accepted(self, tmp_path: Path) -> None:
        from bci_eeg_to_text.hf_model import load_any_backbone

        from transformers import BertConfig, BertModel

        BertModel(
            BertConfig(
                vocab_size=64, hidden_size=32, num_hidden_layers=1,
                num_attention_heads=4, intermediate_size=64,
            )
        ).save_pretrained(tmp_path)
        backbone = load_any_backbone(str(tmp_path), n_channels=8, d_model=32)
        out = backbone(torch.randn(1, 8, 1000))
        assert out.shape[0] == 1

    def test_unknown_folder_raises_actionable_error(self, tmp_path: Path) -> None:
        from bci_eeg_to_text.hf_model import BackboneLoadError, load_any_backbone

        with pytest.raises(BackboneLoadError, match="could not load"):
            load_any_backbone(str(tmp_path / "absent"), n_channels=8, d_model=32)

    def test_loader_passes_trust_remote_code(self, monkeypatch) -> None:
        """Folders with custom architectures require trust_remote_code=True."""
        from bci_eeg_to_text import hf_model

        seen: dict[str, object] = {}

        def fake_from_pretrained(name, **kwargs):
            seen.update(kwargs)
            seen["name"] = name

            class Dummy:
                config = type("C", (), {"hidden_size": 32})()

                def __call__(self, x):
                    return torch.zeros(x.shape[0], 4, 32)

            return Dummy()

        monkeypatch.setattr(hf_model.AutoModel, "from_pretrained", fake_from_pretrained)
        hf_model.load_any_backbone("some/repo", n_channels=8, d_model=32)
        assert seen.get("trust_remote_code") is True
