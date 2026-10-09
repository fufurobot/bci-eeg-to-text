"""Tests for CTC decoding and text metrics."""

from __future__ import annotations

import pytest
import torch

from bci_eeg_to_text.decode import (
    beam_search_decode,
    character_error_rate,
    greedy_decode,
    sentence_accuracy,
)
from bci_eeg_to_text.tokenizer import CharTokenizer

VOCAB = ["我", "想", "你", "好"]


@pytest.fixture
def tokenizer() -> CharTokenizer:
    return CharTokenizer.build(VOCAB)


def logits_from_path(path: list[int], n_classes: int, confidence: float = 0.9) -> torch.Tensor:
    """Build logits whose argmax follows ``path`` exactly."""
    time = len(path)
    logits = torch.full((time, n_classes), (1 - confidence) / (n_classes - 1)).log()
    for t, c in enumerate(path):
        logits[t, c] = torch.log(torch.tensor(confidence))
    return logits


class TestGreedyDecode:
    def test_collapses_repeated_labels(self, tokenizer: CharTokenizer) -> None:
        """CTC must collapse repeats that are not separated by blank."""
        a, b = tokenizer.stoi["我"], tokenizer.stoi["想"]
        logits = logits_from_path([a, a, a, b, b], len(tokenizer)).unsqueeze(0)
        assert greedy_decode(logits, tokenizer) == ["我想"]

    def test_blank_separates_repeated_characters(self, tokenizer: CharTokenizer) -> None:
        """A blank between identical labels means two occurrences."""
        a = tokenizer.stoi["我"]
        logits = logits_from_path([a, 0, a], len(tokenizer)).unsqueeze(0)
        assert greedy_decode(logits, tokenizer) == ["我我"]

    def test_leading_and_trailing_blanks_are_ignored(self, tokenizer: CharTokenizer) -> None:
        a, b = tokenizer.stoi["我"], tokenizer.stoi["想"]
        logits = logits_from_path([0, a, b, 0, 0], len(tokenizer)).unsqueeze(0)
        assert greedy_decode(logits, tokenizer) == ["我想"]

    def test_all_blank_decodes_to_empty_string(self, tokenizer: CharTokenizer) -> None:
        logits = logits_from_path([0, 0, 0], len(tokenizer)).unsqueeze(0)
        assert greedy_decode(logits, tokenizer) == [""]

    def test_batch_decoding(self, tokenizer: CharTokenizer) -> None:
        a, b = tokenizer.stoi["我"], tokenizer.stoi["想"]
        logits = torch.stack(
            [
                logits_from_path([a, 0, b], len(tokenizer)),
                logits_from_path([b, b, 0], len(tokenizer)),
            ]
        )
        assert greedy_decode(logits, tokenizer) == ["我想", "想"]

    def test_input_lengths_mask_padding(self, tokenizer: CharTokenizer) -> None:
        """Frames past input_length are padding and must not be decoded."""
        a, b = tokenizer.stoi["我"], tokenizer.stoi["想"]
        logits = logits_from_path([a, b, b], len(tokenizer)).unsqueeze(0)
        assert greedy_decode(logits, tokenizer, torch.tensor([2])) == ["我想"]
        assert greedy_decode(logits, tokenizer, torch.tensor([3])) == ["我想"]

    def test_rejects_wrong_rank(self, tokenizer: CharTokenizer) -> None:
        with pytest.raises(ValueError, match=r"expected \(batch, time, classes\)"):
            greedy_decode(torch.randn(5, 10), tokenizer)


class TestBeamSearch:
    def test_agrees_with_greedy_on_an_unambiguous_path(self, tokenizer: CharTokenizer) -> None:
        a, b = tokenizer.stoi["我"], tokenizer.stoi["想"]
        logits = logits_from_path([a, 0, b], len(tokenizer), confidence=0.99)
        assert (
            beam_search_decode(logits, tokenizer)[0]
            == greedy_decode(logits.unsqueeze(0), tokenizer)[0]
        )

    def test_recovers_sequence_when_greedy_is_ambiguous(self, tokenizer: CharTokenizer) -> None:
        """Beam search should beat greedy when the best path is not the best prefix."""
        a = tokenizer.stoi["我"]
        time, n = 4, len(tokenizer)
        logits = torch.full((time, n), -10.0)
        logits[:, a] = -0.5
        logits[:, 0] = -0.51
        beam = beam_search_decode(logits, tokenizer, beam_width=8)[0]
        assert beam == "我"

    def test_returns_empty_for_all_blank(self, tokenizer: CharTokenizer) -> None:
        logits = logits_from_path([0, 0, 0], len(tokenizer))
        assert beam_search_decode(logits, tokenizer) == [""]

    def test_rejects_wrong_rank(self, tokenizer: CharTokenizer) -> None:
        with pytest.raises(ValueError, match=r"expected \(time, classes\)"):
            beam_search_decode(torch.randn(2, 5, 10), tokenizer)

    def test_rejects_non_positive_beam_width(self, tokenizer: CharTokenizer) -> None:
        logits = logits_from_path([1, 2], len(tokenizer))
        with pytest.raises(ValueError, match="beam_width must be"):
            beam_search_decode(logits, tokenizer, beam_width=0)

    def test_respects_input_length(self, tokenizer: CharTokenizer) -> None:
        a, b = tokenizer.stoi["我"], tokenizer.stoi["想"]
        logits = logits_from_path([a, b, b, b], len(tokenizer), confidence=0.99)
        assert beam_search_decode(logits, tokenizer, input_length=1)[0] == "我"


class TestCharacterErrorRate:
    def test_identical_strings_have_zero_error(self) -> None:
        assert character_error_rate(["你好"], ["你好"]) == 0.0

    def test_substitution_is_one_edit(self) -> None:
        assert character_error_rate(["你好"], ["你坏"]) == pytest.approx(0.5)

    def test_deletion_and_insertion(self) -> None:
        assert character_error_rate(["你好"], ["你"]) == pytest.approx(0.5)
        assert character_error_rate(["你"], ["你好"]) == pytest.approx(1.0)

    def test_empty_reference_counts_as_error(self) -> None:
        assert character_error_rate([""], ["你"]) == 1.0
        assert character_error_rate([""], [""]) == 0.0

    def test_aggregates_over_corpus(self) -> None:
        # 1 substitution out of 4 characters total.
        assert character_error_rate(["你好", "我想"], ["你坏", "我想"]) == pytest.approx(0.25)

    def test_rejects_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            character_error_rate(["你"], ["你", "好"])


class TestSentenceAccuracy:
    def test_all_correct(self) -> None:
        assert sentence_accuracy(["你好", "我想"], ["你好", "我想"]) == 1.0

    def test_partial_credit_is_per_sentence(self) -> None:
        assert sentence_accuracy(["你好", "我想"], ["你好", "你想"]) == pytest.approx(0.5)

    def test_empty_input_is_zero_not_error(self) -> None:
        assert sentence_accuracy([], []) == 0.0

    def test_rejects_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            sentence_accuracy(["你"], [])
