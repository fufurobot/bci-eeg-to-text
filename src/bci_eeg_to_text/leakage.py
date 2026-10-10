"""Leakage-aware splitting over rows, paragraphs and articles.

Sharing a *recording* across parts is only one way to leak. Two splits can hold
entirely disjoint EEG while sharing text, and a model can then memorise
sentence-to-sentence regularities instead of learning to read the signal.

Four modes, weakest to strongest:

===============  ==================================================================
``none``         nothing enforced; useful as an upper bound
``row``          no identical row appears in two parts (**default**)
``paragraph``    no paragraph appears in two parts
``article``      no chapter appears in two parts (strictest)
===============  ==================================================================

The unit of assignment is the **group**, not the row. Rows belonging to the same
paragraph (or article) are indivisible: splitting them would allow half of a
paragraph in train and half in test, which is precisely the leak the stricter
modes exist to prevent.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from bci_eeg_to_text.splits import SplitError
from bci_eeg_to_text.textstructure import (
    DEFAULT_LEAKAGE_MODE,
    LEAKAGE_MODES,
    RowLabel,
)


class LeakageError(SplitError):
    """Raised when a leakage constraint cannot be satisfied."""


@dataclass(frozen=True)
class TextUnit:
    """One supervised row plus the group that must travel with it."""

    row_id: str
    text: str
    #: Paragraph identity, or None when the row could not be located.
    paragraph_id: str | None
    #: Article identity, or None when the row could not be located.
    article_id: str | None

    def group_for(self, mode: str) -> str:
        """Grouping key for a leakage mode.

        Rows with no recoverable structure fall back to their own row id, so an
        unplaced row can never silently share a group with anything.
        """
        if mode == "none":
            return self.row_id
        if mode == "row":
            return self.row_id
        if mode == "paragraph":
            return self.paragraph_id or self.row_id
        if mode == "article":
            return self.article_id or self.row_id
        raise LeakageError(f"unknown leakage mode {mode!r}; expected one of {LEAKAGE_MODES}")


@dataclass
class LeakageReport:
    """What a leakage check found."""

    mode: str
    checked: int
    violations: list[tuple[str, str, str]] = field(default_factory=list)
    unplaced_rows: int = 0
    n_groups: int = 0

    @property
    def ok(self) -> bool:
        return not self.violations

    def describe(self) -> str:
        if self.mode == "none":
            return "leakage: none (no constraint enforced)"
        state = "clean" if self.ok else f"{len(self.violations)} violations"
        return f"leakage: {self.mode} level, {self.n_groups} groups, {self.checked} rows, {state}"


def units_from_labels(labels: Sequence[RowLabel]) -> list[TextUnit]:
    """Convert row labels into leakage-checkable units."""
    return [
        TextUnit(
            row_id=label.row_id,
            text=label.text,
            paragraph_id=label.paragraph_id,
            article_id=label.article_id,
        )
        for label in labels
    ]


def group_ids(units: Iterable[TextUnit], mode: str) -> set[str]:
    """Distinct groups present in ``units`` for a mode."""
    return {unit.group_for(mode) for unit in units}


def check_leakage(
    parts: dict[str, Sequence[TextUnit]],
    mode: str,
) -> LeakageReport:
    """Verify that no group spans two parts.

    Groups are compared by identity, not by text equality, so a paragraph that
    appears twice in a document is still one group with one owner.
    """
    if mode not in LEAKAGE_MODES:
        raise LeakageError(f"unknown leakage mode {mode!r}; expected one of {LEAKAGE_MODES}")

    report = LeakageReport(mode=mode, checked=sum(len(v) for v in parts.values()))
    report.unplaced_rows = sum(
        1 for units in parts.values() for unit in units if unit.paragraph_id is None
    )
    if mode == "none":
        return report

    owners: dict[str, str] = {}
    for part, units in parts.items():
        for unit in units:
            group = unit.group_for(mode)
            if group in owners and owners[group] != part:
                report.violations.append((group, owners[group], part))
            else:
                owners.setdefault(group, part)
    report.n_groups = len(owners)
    return report


def assign_units_to_parts(
    units: Sequence[TextUnit],
    *,
    mode: str,
    ratios: dict[str, int],
    seed: int = 0,
) -> dict[str, tuple[TextUnit, ...]]:
    """Split text units into parts without breaking a group.

    Groups are allocated whole, largest first, to whichever part is furthest
    below its target. Greedy descending allocation keeps part sizes close to the
    requested ratios while guaranteeing that a group is never divided.
    """
    if mode not in LEAKAGE_MODES:
        raise LeakageError(f"unknown leakage mode {mode!r}; expected one of {LEAKAGE_MODES}")

    grouped: dict[str, list[TextUnit]] = defaultdict(list)
    for unit in units:
        grouped[unit.group_for(mode)].append(unit)

    total = len(units)
    if total == 0:
        return {part: () for part in ratios}

    ratio_total = sum(ratios.values()) or 1
    targets = {part: total * share / ratio_total for part, share in ratios.items()}

    import random

    rng = random.Random(seed)
    # Sort by size, then shuffle within equal sizes for seed variety.
    order = sorted(grouped.items(), key=lambda kv: (-len(kv[1]), rng.random()))

    assigned: dict[str, list[TextUnit]] = {part: [] for part in ratios}
    for _, group_units in order:
        # Ties broken by declaration order so the result is reproducible.
        part = max(
            ratios,
            key=lambda p: (targets[p] - len(assigned[p]), -list(ratios).index(p)),
        )
        assigned[part].extend(group_units)

    return {part: tuple(sorted(group, key=lambda u: u.row_id)) for part, group in assigned.items()}


def summarize_coverage(units: Sequence[TextUnit], mode: str) -> str:
    """Human-readable coverage, so weak structure is visible rather than hidden."""
    if not units:
        return "no text units"
    n_par = sum(1 for u in units if u.paragraph_id)
    n_art = sum(1 for u in units if u.article_id)
    total = len(units)
    return (
        f"{total} rows; {n_par} with a paragraph ({n_par / total:.1%}), "
        f"{n_art} with an article ({n_art / total:.1%}); "
        f"{len(group_ids(units, mode))} {mode}-level groups"
    )


def require_disjoint_groups(
    parts: dict[str, Sequence[TextUnit]],
    mode: str,
) -> None:
    """Raise :class:`LeakageError` if any group spans parts."""
    report = check_leakage(parts, mode)
    if not report.ok:
        group, first, second = report.violations[0]
        raise LeakageError(
            f"{report.mode}-level text leaks across parts: group {group!r} appears "
            f"in both {first!r} and {second!r} "
            f"({len(report.violations)} violations). Choose a stricter --leakage "
            "mode, or accept it explicitly with --leakage none."
        )


def default_mode() -> str:
    """The mode used when the caller does not choose one."""
    return DEFAULT_LEAKAGE_MODE


def describe_mode(mode: str) -> str:
    """One-line explanation of a leakage mode, for CLI help and logs."""
    return {
        "none": "no leakage constraint; text may repeat across parts",
        "row": "no identical row may appear in two parts (default)",
        "paragraph": "no paragraph may appear in two parts",
        "article": "no chapter/document may appear in two parts (strictest)",
    }.get(mode, f"unknown mode {mode!r}")
