"""Tests for the interchangeable CTC backbones.

These cover the contract every backbone must satisfy and, importantly, the CTC
length bookkeeping. Over-reporting output lengths to CTC makes it consume
padding frames as if they were signal, which degrades quietly rather than
raising, so the mapping is asserted against the encoder's real output.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from bci_eeg_to_text.backbones import (
    TRAINING_MODES,
    BackboneConfig,
    BackboneError,
    ConvBackbone,
    CTCModel,
    NeuroProcessor,
    build_backbone,
    build_ctc_model,
    estimate_output_frames,
    resolve_length_map,
)


@pytest.fixture(scope="module")
def tiny_checkpoints() -> dict[str, str]:
    """Locally constructed BERT and GPT-2 checkpoints.

    Built rather than downloaded so the tests run offline and stay fast.
    """
    from transformers import BertConfig, BertModel, GPT2Config, GPT2LMHeadModel

    root = Path(tempfile.mkdtemp())
    bert = root / "bert"
    gpt2 = root / "gpt2"
    if not bert.exists():
        BertModel(
            BertConfig(
                vocab_size=128,
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=64,
                max_position_embeddings=512,
            )
        ).save_pretrained(bert)
    if not gpt2.exists():
        GPT2LMHeadModel(
            GPT2Config(
                vocab_size=128, n_embd=32, n_layer=2, n_head=4, n_positions=128
            )
        ).save_pretrained(gpt2)
    return {"bert": str(bert), "gpt2": str(gpt2)}


class TestLengthMap:
    def test_stride_one_is_identity(self) -> None:
        lengths = torch.tensor([10, 20])
        assert torch.equal(resolve_length_map(lengths, 1), lengths)

    def test_matches_floor_formula(self) -> None:
        assert resolve_length_map(torch.tensor([100]), 8).item() == 13

    def test_estimate_helper_agrees_with_tensor_version(self) -> None:
        for n in (1, 7, 100, 999):
            assert estimate_output_frames(n, 8) == resolve_length_map(
                torch.tensor([n]), 8
            ).item()

    def test_estimate_of_empty_is_zero(self) -> None:
        assert estimate_output_frames(0, 8) == 0


class TestConvBackbone:
    def test_reports_features_not_classes(self) -> None:
        backbone = ConvBackbone(BackboneConfig(n_channels=4, d_model=16))
        out = backbone(torch.randn(2, 4, 2048))
        assert out.shape[0] == 2
        assert out.shape[2] == 16

    def test_length_mapping_is_exact(self) -> None:
        """The reported frames must equal the real frames, at several lengths."""
        backbone = ConvBackbone(BackboneConfig(n_channels=4, d_model=16)).eval()
        for n in (512, 1000, 2048, 4000):
            with torch.no_grad():
                actual = backbone(torch.randn(1, 4, n)).shape[1]
            reported = int(backbone.input_lengths_to_output(torch.tensor([n]))[0])
            assert actual == reported, f"n={n}: {actual} != {reported}"

    def test_rejects_wrong_rank(self) -> None:
        backbone = ConvBackbone(BackboneConfig(n_channels=4, d_model=16))
        with pytest.raises(ValueError, match="expected"):
            backbone(torch.randn(4, 100))


class TestNeuroProcessor:
    def test_produces_expected_width(self) -> None:
        processor = NeuroProcessor(n_channels=4, d_model=16)
        assert processor(torch.randn(2, 4, 2048)).shape[2] == 16

    def test_is_differentiable(self) -> None:
        """Features must stay continuous so CTC can train end to end."""
        processor = NeuroProcessor(n_channels=4, d_model=16)
        out = processor(torch.randn(1, 4, 2048))
        out.sum().backward()
        assert all(p.grad is not None for p in processor.parameters())

    def test_no_double_application_of_blocks(self) -> None:
        """A block must be applied once per call.

        An earlier draft called each block twice while testing the residual
        length, which silently doubled the depth.
        """
        processor = NeuroProcessor(n_channels=4, d_model=16, n_blocks=2).eval()
        calls: list[int] = []
        for index, block in enumerate(processor.blocks):
            original = block.forward

            def counted(x, _orig=original, _index=index):
                calls.append(_index)
                return _orig(x)

            block.forward = counted  # type: ignore[method-assign]
        with torch.no_grad():
            processor(torch.randn(1, 4, 2048))
        assert len(calls) == 2, f"expected 2 block calls, saw {len(calls)}"
        assert calls == [0, 1], "blocks must run once each, in order"


class TestBuildBackbone:
    def test_unknown_family_raises(self) -> None:
        with pytest.raises(BackboneError, match="unknown backbone family"):
            build_backbone("nope", BackboneConfig())

    def test_scratch_aliases_conv(self) -> None:
        assert isinstance(build_backbone("scratch", BackboneConfig()), ConvBackbone)


class TestBuildCTCModel:
    def test_rejects_unknown_mode(self) -> None:
        with pytest.raises(BackboneError, match="unknown mode"):
            build_ctc_model("bogus", n_channels=4, n_classes=10)

    @pytest.mark.parametrize("mode", TRAINING_MODES)
    def test_every_mode_builds_and_runs(self, mode: str, tiny_checkpoints) -> None:
        kwargs: dict[str, object] = {"d_model": 32}
        if mode == "bert-ctc":
            kwargs["pretrained"] = tiny_checkpoints["bert"]
        elif mode.startswith("neurolm"):
            kwargs["extra"] = {
                "lm_name": tiny_checkpoints["gpt2"],
                "processor": "scratch",
                "eeg_tokens": 128,
            }
        model = build_ctc_model(mode, n_channels=4, n_classes=17, **kwargs).eval()
        with torch.no_grad():
            out = model(torch.randn(2, 4, 2048))
        assert out.ndim == 3
        assert out.shape[0] == 2
        assert out.shape[2] == 17

    def test_scratch_and_conv_agree_on_shapes(self) -> None:
        a = build_ctc_model("scratch", n_channels=4, n_classes=9, d_model=16)
        out = a(torch.randn(1, 4, 2048))
        assert out.shape[2] == 9

    def test_length_mapping_feeds_ctc(self, tiny_checkpoints) -> None:
        for mode, kwargs in [
            ("scratch", {}),
            ("bert-ctc", {"pretrained": tiny_checkpoints["bert"]}),
        ]:
            model = build_ctc_model(
                mode, n_channels=4, n_classes=12, d_model=32, **kwargs
            ).eval()
            n = 2048
            with torch.no_grad():
                frames = model(torch.randn(1, 4, n)).shape[1]
            reported = int(model.input_lengths_to_output(torch.tensor([n]))[0])
            assert reported <= frames, f"{mode}: reported {reported} > actual {frames}"


class TestFreezing:
    def test_frozen_backbone_trains_far_fewer_parameters(self, tiny_checkpoints) -> None:
        """Freezing is the point of the BERT mode: the head alone should train."""
        frozen = build_ctc_model(
            "bert-ctc",
            n_channels=4,
            n_classes=12,
            d_model=32,
            pretrained=tiny_checkpoints["bert"],
            freeze_backbone=True,
        )
        full = build_ctc_model(
            "bert-ctc",
            n_channels=4,
            n_classes=12,
            d_model=32,
            pretrained=tiny_checkpoints["bert"],
            freeze_backbone=False,
        )
        assert frozen.trainable_parameter_count() < full.trainable_parameter_count()
        summary = frozen.parameter_summary()
        assert summary["frozen"] > 0
        assert 0 < summary["trainable_fraction"] < 1

    def test_frozen_parameters_receive_no_gradients(self, tiny_checkpoints) -> None:
        model = build_ctc_model(
            "bert-ctc",
            n_channels=4,
            n_classes=12,
            d_model=32,
            pretrained=tiny_checkpoints["bert"],
            freeze_backbone=True,
        )
        model(torch.randn(1, 4, 2048)).sum().backward()
        assert all(
            p.grad is None for p in model.backbone.bert.parameters()
        ), "frozen BERT weights must not accumulate gradients"
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.backbone.adapter.parameters()
        ), "the adapter must still learn"

    def test_trainable_parameters_exclude_frozen(self, tiny_checkpoints) -> None:
        model = build_ctc_model(
            "bert-ctc",
            n_channels=4,
            n_classes=12,
            d_model=32,
            pretrained=tiny_checkpoints["bert"],
            freeze_backbone=True,
        )
        assert all(p.requires_grad for p in model.trainable_parameters())
        assert len(model.trainable_parameters()) < len(list(model.parameters()))


class TestBertLongSequences:
    def test_adaptive_pooling_keeps_long_trials_within_limit(
        self, tiny_checkpoints
    ) -> None:
        """BERT has a fixed context; long EEG trials must be pooled, not crash."""
        model = build_ctc_model(
            "bert-ctc",
            n_channels=4,
            n_classes=8,
            d_model=32,
            pretrained=tiny_checkpoints["bert"],
        ).eval()
        limit = model.backbone.max_frames
        for n in (1135, 8000, 64000):
            with torch.no_grad():
                frames = model(torch.randn(1, 4, n)).shape[1]
            assert frames <= limit, f"n={n}: {frames} frames exceeds {limit}"

    def test_length_mapping_still_exact_with_pooling(self, tiny_checkpoints) -> None:
        model = build_ctc_model(
            "bert-ctc",
            n_channels=4,
            n_classes=8,
            d_model=32,
            pretrained=tiny_checkpoints["bert"],
        ).eval()
        for n in (800, 1135, 8000, 64000):
            with torch.no_grad():
                actual = model(torch.randn(1, 4, n)).shape[1]
            reported = int(model.input_lengths_to_output(torch.tensor([n]))[0])
            assert actual == reported, f"n={n}: {actual} != {reported}"

    def test_short_trials_are_not_pooled(self, tiny_checkpoints) -> None:
        model = build_ctc_model(
            "bert-ctc",
            n_channels=4,
            n_classes=8,
            d_model=32,
            pretrained=tiny_checkpoints["bert"],
        ).eval()
        assert model.backbone._pool_factor_for(100) == 1

    def test_long_trials_get_a_pooling_factor(self, tiny_checkpoints) -> None:
        model = build_ctc_model(
            "bert-ctc",
            n_channels=4,
            n_classes=8,
            d_model=32,
            pretrained=tiny_checkpoints["bert"],
        ).eval()
        assert model.backbone._pool_factor_for(4096) > 1


class TestMissingCheckpoint:
    def test_bad_bert_path_raises_with_guidance(self) -> None:
        with pytest.raises(BackboneError, match="could not load BERT checkpoint"):
            build_ctc_model(
                "bert-ctc",
                n_channels=4,
                n_classes=8,
                d_model=16,
                pretrained="/nonexistent/definitely/not/here",
            )

    def test_bad_lm_path_raises_with_guidance(self) -> None:
        with pytest.raises(BackboneError, match="could not load causal LM"):
            build_ctc_model(
                "neurolm-adapter",
                n_channels=4,
                n_classes=8,
                d_model=16,
                extra={"lm_name": "/nonexistent/lm"},
            )


class TestCTCModelWrapper:
    def test_forward_shape_and_head(self) -> None:
        backbone = ConvBackbone(BackboneConfig(n_channels=4, d_model=16))
        model = CTCModel(backbone, n_classes=23)
        assert model(torch.randn(2, 4, 2048)).shape[2] == 23

    def test_parameter_summary_adds_up(self) -> None:
        backbone = ConvBackbone(BackboneConfig(n_channels=4, d_model=16))
        summary = CTCModel(backbone, n_classes=10).parameter_summary()
        assert summary["total"] == summary["trainable"] + summary["frozen"]
        assert summary["trainable_fraction"] == pytest.approx(1.0)
