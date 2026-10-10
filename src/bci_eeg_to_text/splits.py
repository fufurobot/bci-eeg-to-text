"""Train/valid/test splitting at the participant level.

EEG recordings are highly subject-specific: a model trained and evaluated on
different runs of the *same* person is mostly measuring how well it memorised
that person, not how well EEG-to-text generalises. The default here is therefore
splitting by **subject**.

Two additional requirements shape this module:

* **Sex balance.** With few subjects (ds005170 has only 4 usable), an unlucky
  random split can put every female subject in one side, so a model could
  exploit sex-linked signal structure. Splits aim to balance sex across parts.
* **Explicit run-level splitting.** Within-participant splits are legitimate for
  some questions (e.g. session-to-session generalisation) but must never be
  produced by accident, so they require an explicit level and are flagged.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field


class SplitError(ValueError):
    """Raised when a requested split cannot be satisfied."""


@dataclass(frozen=True)
class Participant:
    """Minimal participant metadata needed for balanced splitting."""

    #: Globally unique id, namespaced by dataset when merging (e.g. "chisco:sub-01").
    subject_id: str
    #: Dataset the participant came from, or None for single-dataset use.
    dataset: str | None = None
    #: Reported sex ("F"/"M"), used for balance. None disables balancing.
    sex: str | None = None

    @property
    def short_id(self) -> str:
        """Subject id without the dataset namespace."""
        return self.subject_id.split(":", 1)[-1]


@dataclass(frozen=True)
class Split:
    """A partition of subjects into named parts."""

    parts: Mapping[str, tuple[str, ...]]
    #: True when the split separates subjects (the safe default).
    subject_level: bool = True
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def subjects(self, part: str) -> tuple[str, ...]:
        return self.parts[part]

    @property
    def n_subjects(self) -> int:
        return sum(len(v) for v in self.parts.values())

    def sex_balance(self, participants: Mapping[str, Participant]) -> dict[str, dict[str, int]]:
        """Count participants of each sex per part."""
        out: dict[str, dict[str, int]] = {}
        for part, subjects in self.parts.items():
            counts: dict[str, int] = defaultdict(int)
            for subject in subjects:
                counts[participants[subject].sex or "unknown"] += 1
            out[part] = dict(counts)
        return out


DEFAULT_RATIOS = {"train": 3, "valid": 1, "test": 1}


def _target_counts(n_subjects: int, ratios: Mapping[str, int]) -> dict[str, int]:
    """Largest-remainder allocation of ``n_subjects`` across parts."""
    total = sum(ratios.values())
    if total <= 0:
        raise SplitError("ratios must sum to a positive number")
    exact = {p: n_subjects * r / total for p, r in ratios.items()}
    counts = {p: int(v) for p, v in exact.items()}
    # Distribute the remainder to the largest fractional parts, ties broken by
    # the declared order so the result is deterministic.
    order = list(ratios)
    remainder = n_subjects - sum(counts.values())
    for part in sorted(order, key=lambda p: (-(exact[p] - counts[p]), order.index(p)))[:remainder]:
        counts[part] += 1
    return counts


def make_splits(
    participants: Sequence[Participant],
    *,
    ratios: Mapping[str, int] | None = None,
    level: str = "subject",
    balance: str | None = "sex",
    seed: int = 0,
    runs_per_subject: Mapping[str, Sequence[str]] | None = None,
) -> Split:
    """Partition participants into train/valid/test.

    ``level="subject"`` (default) keeps every subject wholly inside one part, so
    the result estimates cross-subject generalisation. ``level="run"`` splits
    runs instead and is intended only for within-participant experiments; it
    attaches a warning because such scores are not cross-subject estimates.
    """
    if ratios is None:
        ratios = dict(DEFAULT_RATIOS)
    if not participants:
        raise SplitError("no participants to split")
    if level not in {"subject", "run"}:
        raise SplitError(f"level must be 'subject' or 'run', got {level!r}")
    if balance not in {None, "sex"}:
        raise SplitError(f"balance must be None or 'sex', got {balance!r}")

    by_id = {p.subject_id: p for p in participants}
    if len(by_id) != len(participants):
        raise SplitError("duplicate subject ids in participants")
    if level == "subject":
        return _split_by_subject(by_id, ratios, balance, seed)

    warnings = (
        "split level is 'run': parts share participants, so the reported score "
        "measures within-participant performance and is NOT a cross-subject "
        "estimate. Use level='subject' for generalisation claims.",
    )
    return _split_by_run(by_id, ratios, balance, seed, runs_per_subject, warnings)


def _split_by_subject(
    by_id: Mapping[str, Participant],
    ratios: Mapping[str, int],
    balance: str | None,
    seed: int,
) -> Split:
    counts = _target_counts(len(by_id), ratios)
    parts = list(ratios)

    # Group by sex so each part can be filled with a balanced quota. Shuffling
    # within a group avoids always putting (say) the lowest ids in train.
    rng = random.Random(seed)
    if balance == "sex":
        groups: dict[str, list[str]] = defaultdict(list)
        for pid in sorted(by_id):
            groups[by_id[pid].sex or "unknown"].append(pid)
        for ids in groups.values():
            rng.shuffle(ids)
    else:
        ids = sorted(by_id)
        rng.shuffle(ids)
        groups = {"all": ids}

    assignment: dict[str, list[str]] = {p: [] for p in parts}

    # Allocate part-by-part, and within each part take from every sex group in
    # turn. Filling part-first (rather than subject-first) is what keeps sex
    # balanced: each part receives its quota drawn evenly across groups, instead
    # of whichever group happened to be processed first.
    for part in parts:
        quota = counts[part]
        while len(assignment[part]) < quota:
            # Prefer the sex group with the most subjects still unassigned, so
            # a small group is not exhausted early and left unbalanced.
            candidates = [(len(ids), sex) for sex, ids in groups.items() if ids]
            if not candidates:
                break
            _, sex = max(candidates, key=lambda kv: kv[0])
            assignment[part].append(groups[sex].pop())

    # Any remainder (more parts' quotas than subjects, or uneven groups) goes to
    # whichever group is largest, still preferring under-filled parts.
    leftovers = [pid for ids in groups.values() for pid in ids]
    for pid in leftovers:
        smallest = min(parts, key=lambda p: (len(assignment[p]), parts.index(p)))
        assignment[smallest].append(pid)

    return Split(parts={p: tuple(sorted(v)) for p, v in assignment.items()})


def _split_by_run(
    by_id: Mapping[str, Participant],
    ratios: Mapping[str, int],
    balance: str | None,
    seed: int,
    runs_per_subject: Mapping[str, Sequence[str]] | None,
    warnings: tuple[str, ...],
) -> Split:
    if not runs_per_subject:
        raise SplitError("level='run' requires runs_per_subject")
    rng = random.Random(seed)
    parts = list(ratios)
    assignment: dict[str, list[str]] = {p: [] for p in parts}

    for pid in sorted(by_id):
        run_ids = sorted(runs_per_subject.get(pid, ()))
        if not run_ids:
            continue
        rng.shuffle(run_ids)
        counts = _target_counts(len(run_ids), ratios)
        cursor = 0
        for part in parts:
            take = counts[part]
            for run in run_ids[cursor : cursor + take]:
                assignment[part].append(f"{pid}/{run}")
            cursor += take

    return Split(
        parts={p: tuple(sorted(v)) for p, v in assignment.items()},
        subject_level=False,
        warnings=warnings,
    )


def validate_subject_disjoint(
    split: Split, participants: Mapping[str, Participant] | Iterable[str]
) -> None:
    """Assert that no subject appears in more than one part.

    Only meaningful for subject-level splits. A run-level split deliberately
    shares participants across parts, so applying this check there would reject
    every valid split; the caller opts in by asking for subject level.
    """
    if not split.subject_level:
        return
    seen: dict[str, str] = {}
    for part, entries in split.parts.items():
        for entry in entries:
            if entry in seen:
                raise SplitError(
                    f"subject {entry!r} appears in both {seen[entry]!r} and {part!r}; "
                    "this leaks participant identity across parts"
                )
            seen[entry] = part
