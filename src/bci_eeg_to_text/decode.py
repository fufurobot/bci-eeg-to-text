"""CTC decoding and text metrics.

Two decoders are provided:

* :func:`greedy_decode` -- collapse the argmax path. Cheap, used during
  training to watch progress.
* :func:`beam_search_decode` -- prefix beam search with a length-normalized
  score. More accurate, used for reported results.

Both must agree on the CTC collapsing rule: repeated labels collapse, and blank
separates repeats. Getting this wrong produces text that looks almost right,
which is why the tests pin the edge cases explicitly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from bci_eeg_to_text.tokenizer import CharTokenizer


def _dedupe_and_strip_blank(path: Sequence[int], blank: int) -> list[int]:
    """Apply the CTC collapse rule to a single alignment path."""
    collapsed: list[int] = []
    previous: int | None = None
    for token in path:
        token = int(token)
        if token != previous and token != blank:
            collapsed.append(token)
        previous = token
    return collapsed


def greedy_decode(
    logits: torch.Tensor,
    tokenizer: CharTokenizer,
    input_lengths: torch.Tensor | None = None,
) -> list[str]:
    """Decode ``(batch, time, classes)`` logits by collapsing the argmax path."""
    if logits.ndim != 3:
        raise ValueError(f"expected (batch, time, classes), got {tuple(logits.shape)}")
    batch, time, _ = logits.shape
    lengths = [time] * batch if input_lengths is None else [int(v) for v in input_lengths]

    indices = logits.argmax(dim=-1)
    results: list[str] = []
    for i in range(batch):
        n = max(0, min(lengths[i], time))
        path = indices[i, :n].tolist()
        results.append(tokenizer.decode(_dedupe_and_strip_blank(path, tokenizer.blank)))
    return results


@dataclass(frozen=True)
class Beam:
    """One beam-search hypothesis: the collapsed prefix and its log-probability."""

    tokens: tuple[int, ...]
    log_prob_blank: float
    log_prob_non_blank: float

    @property
    def log_prob(self) -> float:
        return float(
            torch.logaddexp(
                torch.tensor(self.log_prob_blank), torch.tensor(self.log_prob_non_blank)
            )
        )


def _normalized_score(beam: Beam) -> float:
    """Length-normalized log-probability, used for both pruning and selection.

    Shorter hypotheses are not penalized for having fewer terms in their sum.
    """
    return beam.log_prob / max(1, len(beam.tokens))


def _accumulate(
    candidates: dict[tuple[int, ...], list[float]],
    prefix: tuple[int, ...],
    p_blank: float,
    p_non_blank: float,
    neg_inf: float,
) -> None:
    """Add a path probability into ``candidates[prefix]`` in log space.

    ``candidates`` maps a prefix to ``[log P(prefix, ends in blank),
    log P(prefix, ends in non-blank)]``, which is the sufficient statistic for
    CTC prefix beam search.
    """
    slot = candidates.setdefault(prefix, [neg_inf, neg_inf])
    slot[0] = float(torch.logaddexp(torch.tensor(slot[0]), torch.tensor(p_blank)))
    slot[1] = float(torch.logaddexp(torch.tensor(slot[1]), torch.tensor(p_non_blank)))


def beam_search_decode(
    logits: torch.Tensor,
    tokenizer: CharTokenizer,
    *,
    beam_width: int = 8,
    input_length: int | None = None,
    normalize_by_length: bool = False,
) -> list[str]:
    """Prefix beam search over the CTC lattice for a single utterance.

    ``logits`` must be ``(time, classes)``. Beams are scored with
    ``log P(prefix | x)`` accumulated over both the blank-terminated and
    non-blank-terminated paths, which is the standard CTC prefix probability.

    Beams are pruned by raw prefix probability, and the winner maximizes total
    posterior unless ``normalize_by_length`` is set.
    """
    if logits.ndim != 2:
        raise ValueError(f"expected (time, classes), got {tuple(logits.shape)}")
    if beam_width < 1:
        raise ValueError(f"beam_width must be >= 1, got {beam_width}")

    time = logits.shape[0] if input_length is None else min(input_length, logits.shape[0])
    log_probs = torch.log_softmax(logits[:time].float(), dim=-1)
    blank = tokenizer.blank
    neg_inf = float("-inf")

    # Empty prefix: probability 1 of being blank-terminated, 0 otherwise.
    beams: dict[tuple[int, ...], Beam] = {
        (): Beam(tokens=(), log_prob_blank=0.0, log_prob_non_blank=neg_inf)
    }

    for t in range(time):
        step = log_probs[t]
        candidates: dict[tuple[int, ...], list[float]] = {}

        # Prune by raw prefix probability, which is the standard beam-search
        # objective: it keeps the prefixes that carry the most posterior mass
        # and will accumulate the most as further frames are consumed. Pruning
        # by the length-normalized score instead discards short prefixes while
        # they still have time to grow, which measurably degrades the result.
        ordered = sorted(beams.items(), key=lambda kv: kv[1].log_prob, reverse=True)[:beam_width]

        for prefix, beam in ordered:
            # Extending with blank leaves the prefix unchanged. Emitting blank
            # can only produce a blank-terminated path, so BOTH previous endings
            # collapse into the blank slot and the non-blank slot gets nothing:
            #   p_b(l, t) = (p_b(l, t-1) + p_nb(l, t-1)) * P(blank)
            # Routing p_nb into the non-blank slot here loses probability mass
            # and silently drops characters (e.g. "A blank A" decodes as "A").
            _accumulate(candidates, prefix, beam.log_prob + float(step[blank]), neg_inf, neg_inf)

            for c in range(step.shape[-1]):
                if c == blank:
                    continue
                p = float(step[c])
                if c == (prefix[-1] if prefix else None):
                    # Repeat of the last character: the non-blank path keeps the
                    # same prefix, the blank path appends a new occurrence.
                    _accumulate(candidates, prefix, neg_inf, beam.log_prob_non_blank + p, neg_inf)
                    _accumulate(candidates, (*prefix, c), neg_inf, beam.log_prob_blank + p, neg_inf)
                else:
                    _accumulate(candidates, (*prefix, c), neg_inf, beam.log_prob + p, neg_inf)

        beams = {
            prefix: Beam(tokens=prefix, log_prob_blank=vals[0], log_prob_non_blank=vals[1])
            for prefix, vals in candidates.items()
        }

    if not beams:
        return [""]

    # CTC decoding is defined as maximizing the total posterior P(l | x), so
    # select on raw log-probability. Length normalization is available as an
    # explicit opt-in because it biases towards shorter output, which can help
    # when a model emits fragmented text but is not the maximum-posterior
    # answer.
    if normalize_by_length:
        best = max(beams.values(), key=_normalized_score)
    else:
        best = max(beams.values(), key=lambda beam: beam.log_prob)
    return [tokenizer.decode(list(best.tokens))]


def character_error_rate(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Levenshtein distance over characters, normalized by reference length."""
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have equal length")
    total_distance = 0
    total_length = 0
    for ref, hyp in zip(references, hypotheses, strict=True):
        total_distance += _levenshtein(ref, hyp)
        total_length += len(ref)
    if total_length == 0:
        return 0.0 if total_distance == 0 else 1.0
    return total_distance / total_length


def sentence_accuracy(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Fraction of sentences decoded exactly correctly."""
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have equal length")
    if not references:
        return 0.0
    correct = sum(1 for r, h in zip(references, hypotheses, strict=True) if r == h)
    return correct / len(references)


def _levenshtein(a: str, b: str) -> int:
    """Edit distance between two strings, using O(min(n, m)) memory."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + (ca != cb),  # substitution
                )
            )
        previous = current
    return previous[-1]
