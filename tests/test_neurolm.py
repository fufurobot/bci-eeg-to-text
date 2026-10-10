"""Tests for loading NeuroLM components.

NeuroLM weights are not vendored and cannot be downloaded in CI, so these cover
the resolution logic and the failure modes, which is what actually goes wrong in
practice: a missing checkpoint or a missing implementation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from bci_eeg_to_text.neurolm import (
    NEUROLM_HF_REPO,
    NEUROLM_VARIANTS,
    NeuroLMError,
    describe_neuro_processor,
    load_neuro_processor,
    repo_root,
)


class TestRepoRoot:
    def test_returns_none_when_no_checkout_exists(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("NEUROLM_ROOT", str(tmp_path / "absent"))
        monkeypatch.chdir(tmp_path)
        assert repo_root() is None

    def test_honours_the_environment_variable(self, tmp_path, monkeypatch) -> None:
        fake = tmp_path / "NeuroLM"
        (fake / "model").mkdir(parents=True)
        (fake / "model" / "model_neurolm.py").write_text("# stub")
        monkeypatch.setenv("NEUROLM_ROOT", str(fake))
        assert repo_root() == fake

    def test_finds_a_sibling_checkout(self, tmp_path, monkeypatch) -> None:
        fake = tmp_path / "NeuroLM"
        (fake / "model").mkdir(parents=True)
        (fake / "model" / "model_neurolm.py").write_text("# stub")
        monkeypatch.delenv("NEUROLM_ROOT", raising=False)
        monkeypatch.chdir(tmp_path)
        assert repo_root() == Path("NeuroLM")


class TestLoadNeuroProcessor:
    def test_missing_checkpoint_returns_untrained_processor(self) -> None:
        """The adapter mode must still be runnable end to end without weights."""
        processor = load_neuro_processor(None, n_channels=8, d_model=16)
        # Long enough that the strided blocks still leave frames after pooling.
        out = processor(torch.randn(1, 8, 4096))
        assert out.shape[0] == 1
        assert "untrained" in describe_neuro_processor(processor)

    def test_absent_path_raises_with_download_hint(self, tmp_path) -> None:
        with pytest.raises(NeuroLMError, match="checkpoint not found") as exc:
            load_neuro_processor(str(tmp_path / "nope.pt"), n_channels=8, d_model=16)
        assert NEUROLM_HF_REPO in str(exc.value)

    def test_existing_checkpoint_without_implementation_raises(self, tmp_path, monkeypatch) -> None:
        """Silently training on random features would be far worse than failing."""
        import bci_eeg_to_text.neurolm as module

        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name.endswith("neurolm_native"):
                raise ImportError("simulated absence")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", fake_import)
        path = tmp_path / "VQ.pt"
        torch.save({"model": {"a": torch.zeros(1)}}, path)
        with pytest.raises(NeuroLMError, match="no implementation to load it into"):
            module.load_neuro_processor(str(path), n_channels=8, d_model=16)

    def test_loads_weights_into_the_native_implementation(self, tmp_path) -> None:
        from bci_eeg_to_text.neurolm_native import NeuralTransformerVQ, VQConfig

        config = VQConfig(
            n_layer=1,
            n_head=4,
            n_embd=32,
            block_size=8,
            patch_size=50,
            codebook_size=16,
            codebook_dim=8,
        )
        source = NeuralTransformerVQ(n_channels=8, config=config)
        # Checkpoints use the upstream 'VQ.encoder.' prefix.
        state = {f"VQ.encoder.{k}": v for k, v in source.state_dict().items()}
        path = tmp_path / "VQ.pt"
        torch.save({"model": state}, path)

        loaded = load_neuro_processor(str(path), n_channels=8, d_model=8)
        assert all(not p.requires_grad for p in loaded.parameters()), "processor is frozen"
        missing = getattr(loaded, "load_metadata", {}).get("missing_keys", -1)
        assert missing == 0, f"{missing} weights failed to map"


class TestDescribe:
    def test_reports_source_and_state(self) -> None:
        processor = load_neuro_processor(None, n_channels=8, d_model=16)
        description = describe_neuro_processor(processor)
        assert "params" in description
        assert "trainable" in description

    def test_reports_frozen_when_frozen(self, tmp_path) -> None:
        from bci_eeg_to_text.neurolm_native import NeuralTransformerVQ, VQConfig

        config = VQConfig(
            n_layer=1, n_head=4, n_embd=32, block_size=8, codebook_size=8, codebook_dim=8
        )
        source = NeuralTransformerVQ(n_channels=8, config=config)
        path = tmp_path / "VQ.pt"
        torch.save({"model": {f"VQ.encoder.{k}": v for k, v in source.state_dict().items()}}, path)
        assert "frozen" in describe_neuro_processor(load_neuro_processor(str(path), 8, 8))


class TestVariants:
    def test_published_variants_are_listed(self) -> None:
        assert "NeuroLM-B" in NEUROLM_VARIANTS
        assert len(NEUROLM_VARIANTS) == 3
