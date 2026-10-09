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


def exact_ctc_distribution(
    logits: torch.Tensor, tokenizer: CharTokenizer
) -> dict[tuple[int, ...], float]:
    """Exact CTC posterior over collapsed label sequences by path enumeration.

    Enumerates every alignment of length T, collapses it with the CTC rule and
    sums the probabilities. Exponential, so tiny inputs only -- but this is the
    ground truth that beam search must reproduce.

    Keyed by the collapsed **id tuple**, never by decoded text: ``decode`` drops
    UNK, so distinct sequences such as ``(想,)`` and ``(UNK, 想)`` would merge
    and inflate each other's mass, making a correct decoder look wrong.
    """
    import itertools
    import math
    from collections import defaultdict

    log_probs = torch.log_softmax(logits.float(), dim=-1)
    time, n_classes = log_probs.shape
    blank = tokenizer.blank
    total: dict[tuple[int, ...], float] = defaultdict(float)
    for path in itertools.product(range(n_classes), repeat=time):
        log_p = sum(float(log_probs[t, path[t]]) for t in range(time))
        collapsed: list[int] = []
        previous: int | None = None
        for token in path:
            if token != previous and token != blank:
                collapsed.append(token)
            previous = token
        total[tuple(collapsed)] += math.exp(log_p)
    return dict(total)


def _as_text(tokenizer: CharTokenizer, ids: tuple[int, ...]) -> str:
    """Render a collapsed id tuple as text for readable assertions."""
    return tokenizer.decode(list(ids))


class TestBeamSearchCorrectness:
    """Regression tests for a blank-transition bug that dropped characters."""

    def test_blank_can_separate_a_doubled_character(self, tokenizer: CharTokenizer) -> None:
        """A blank between identical labels must be ABLE to yield two chars.

        This does not assert that the doubled output wins -- the maximum
        posterior may still be the single character -- only that the doubled
        hypothesis retains the mass coming through the blank path. Before the
        blank-transition fix it was unreachable.
        """
        a = tokenizer.stoi["我"]
        logits = torch.zeros(3, len(tokenizer))
        logits[0, a] = 3.0
        logits[2, a] = 3.0
        exact = exact_ctc_distribution(logits, tokenizer)

        # The doubled hypothesis must carry real mass, roughly 5/6 of the single
        # one here, rather than collapsing to ~0.
        doubled = exact[(a, a)]
        assert doubled > 0.1, f"doubled hypothesis lost its mass: {doubled}"

        # And beam search must agree with exact about the winner.
        expected = _as_text(tokenizer, max(exact, key=lambda k: exact[k]))
        got = beam_search_decode(logits, tokenizer, beam_width=8)[0]
        assert got == expected

    def test_blank_separated_repeat_wins_when_it_is_the_best(
        self, tokenizer: CharTokenizer
    ) -> None:
        """When the middle frame strongly favours blank, two chars must win."""
        a = tokenizer.stoi["我"]
        logits = torch.full((3, len(tokenizer)), -6.0)
        logits[0, a] = 3.0
        logits[1, 0] = 3.0
        logits[2, a] = 3.0
        exact = exact_ctc_distribution(logits, tokenizer)
        expected = _as_text(tokenizer, max(exact, key=lambda k: exact[k]))
        assert expected == "我我", f"expected doubled output, exact best was {expected!r}"
        assert beam_search_decode(logits, tokenizer, beam_width=8)[0] == "我我"

    def test_matches_exact_argmax(self, tokenizer: CharTokenizer) -> None:
        """With a wide beam, beam search must find the exact CTC argmax.

        CTC decoding maximizes the total posterior P(l | x), so that is the
        reference here. The comparison is on decoded *text*: ``decode`` drops
        UNK, so distinct id sequences can render identically and re-encoding
        the output is not a faithful inverse.
        """
        torch.manual_seed(0)
        for trial in range(40):
            logits = torch.randn(4, len(tokenizer)) * 2.0
            exact = exact_ctc_distribution(logits, tokenizer)
            best_text = _as_text(tokenizer, max(exact, key=lambda k: exact[k]))
            got = beam_search_decode(logits, tokenizer, beam_width=512)[0]
            assert got == best_text, (
                f"trial {trial}: got {got!r}, exact argmax {best_text!r}, logits={logits.tolist()}"
            )

    def test_length_normalization_is_opt_in(self, tokenizer: CharTokenizer) -> None:
        """Normalizing favours shorter output, so it must not change the default."""
        torch.manual_seed(0)
        logits = torch.randn(4, len(tokenizer)) * 2.0
        default = beam_search_decode(logits, tokenizer, beam_width=512)[0]
        normalized = beam_search_decode(
            logits, tokenizer, beam_width=512, normalize_by_length=True
        )[0]
        assert len(normalized) <= len(default) + 1

    def test_blank_transition_conserves_mass(self, tokenizer: CharTokenizer) -> None:
        """The posterior must sum to 1, so no probability is misrouted."""
        torch.manual_seed(1)
        logits = torch.randn(3, len(tokenizer))
        exact = exact_ctc_distribution(logits, tokenizer)
        assert abs(sum(exact.values()) - 1.0) < 1e-6

    def test_does_not_drop_characters_on_uncertain_input(self, tokenizer: CharTokenizer) -> None:
        """The original bug returned a prefix with far less mass than the truth."""
        torch.manual_seed(0)
        logits = torch.randn(4, len(tokenizer)) * 2.0
        exact = exact_ctc_distribution(logits, tokenizer)
        got = beam_search_decode(logits, tokenizer, beam_width=512)[0]
        assert got == _as_text(tokenizer, max(exact, key=lambda k: exact[k]))

    def test_repeat_branch_matches_exact_on_blank_separated_pairs(
        self, tokenizer: CharTokenizer
    ) -> None:
        """Systematic check of the repeat/blank interaction against exact truth."""
        a = tokenizer.stoi["我"]
        for p_char in (0.2, 0.5, 0.9):
            logits = torch.full((3, len(tokenizer)), 0.01)
            logits[0, a] = p_char
            logits[1, 0] = p_char
            logits[2, a] = p_char
            exact = exact_ctc_distribution(logits, tokenizer)
            expected = _as_text(tokenizer, max(exact, key=lambda k: exact[k]))
            got = beam_search_decode(logits, tokenizer, beam_width=64)[0]
            assert got == expected, f"p_char={p_char}: got {got!r}, exact {expected!r}"
