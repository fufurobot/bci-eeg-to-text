"""Leakage as a two-axis matrix, reported explicitly rather than silently.

A split can leak along several independent axes, and a good-looking score can
come from any of them:

**Subject axis** -- who was recorded.

* ``none``           no constraint; participants may be shared across parts
* ``one_subject``    all parts come from a single participant (the strongest
                     within-participant setting; generalisation is not measured)
* ``cross_subject``  no participant appears in two parts (**default**)

**Text axis** -- what was read.

* ``none``           no constraint
* ``row``            no identical row in two parts (**default**)
* ``paragraph``      no paragraph in two parts
* ``article``        no chapter in two parts (strictest)

**Sex axis** -- balance of the ``sex`` field in ``participants.tsv``. Any
``|F - M|`` difference inside a part is reported, because a part made up
entirely of one sex lets a model exploit sex-linked signal structure.

**Run axis** -- a single recording appearing in two parts.

Every axis is always reported. Silence is dangerous here: "no output" is
indistinguishable from "no leakage", so the report names what is enforced, what
is **not** enforced, and exactly which axis leaks and by how much.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from bci_eeg_to_text.splits import Participant, SplitError
from bci_eeg_to_text.textstructure import DEFAULT_LEAKAGE_MODE, LEAKAGE_MODES

#: Subject-axis policies, weakest to strongest.
SUBJECT_MODES = ("none", "one_subject", "cross_subject")

#: Default subject policy: participants never cross a part boundary.
DEFAULT_SUBJECT_MODE = "cross_subject"

#: Text-axis policies (shared with textstructure).
TEXT_MODES = LEAKAGE_MODES

#: Datasets shipping source documents, so paragraph/article modes are meaningful
#: only for these.
DOCUMENT_DATASETS = ("chineseeeg",)


class LeakageError(SplitError):
    """Raised when a leakage constraint cannot be satisfied."""


@dataclass(frozen=True)
class TextUnit:
    """One supervised row plus the group that must travel with it."""

    row_id: str
    text: str
    paragraph_id: str | None
    article_id: str | None
    #: Dataset the row came from, so text modes can be restricted by corpus.
    dataset: str | None = None

    def group_for(self, mode: str) -> str:
        """Grouping key for a text mode.

        Rows with no recoverable structure fall back to their own row id, so an
        unplaced row can never silently share a group with another.
        """
        if mode in {"none", "row"}:
            return self.row_id
        if mode == "paragraph":
            return self.paragraph_id or self.row_id
        if mode == "article":
            return self.article_id or self.row_id
        raise LeakageError(f"unknown text mode {mode!r}; expected one of {TEXT_MODES}")


@dataclass
class AxisReport:
    """Result for one leakage axis."""

    name: str
    mode: str
    enforced: bool
    violations: list[str] = field(default_factory=list)
    detail: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.violations

    def describe(self) -> str:
        """One line naming the mode and, when unenforced, saying so plainly."""
        if not self.enforced:
            return f"{self.name}={self.mode}: NOT ENFORCED (upper bound)"
        if self.ok:
            return f"{self.name}={self.mode}: ok"
        return f"{self.name}={self.mode}: {len(self.violations)} violation(s)"


@dataclass
class LeakageMatrix:
    """Leakage across all axes, always fully reported."""

    subject: AxisReport
    text: AxisReport
    sex: AxisReport
    runs: AxisReport

    @property
    def axes(self) -> tuple[AxisReport, ...]:
        return (self.subject, self.text, self.sex, self.runs)

    @property
    def ok(self) -> bool:
        """True when every *enforced* axis is clean."""
        return all(axis.ok for axis in self.axes if axis.enforced)

    @property
    def leaking_axes(self) -> list[str]:
        """Names of enforced axes that are violated."""
        return [axis.name for axis in self.axes if axis.enforced and not axis.ok]

    @property
    def unenforced_axes(self) -> list[str]:
        return [axis.name for axis in self.axes if not axis.enforced]

    def describe(self, *, max_violations: int = 5) -> str:
        """Multi-line report. Never silent, and always names what leaks."""
        lines = ["leakage matrix:"]
        for axis in self.axes:
            marker = "  " if axis.enforced else "! "
            lines.append(f"{marker}{axis.describe()}")
            for violation in axis.violations[:max_violations]:
                lines.append(f"      - {violation}")
            extra = len(axis.violations) - max_violations
            if extra > 0:
                lines.append(f"      - ... {extra} more")
        if self.unenforced_axes:
            # Say which axes are unchecked, so a reader cannot mistake silence
            # for safety.
            lines.append(
                "  unenforced: "
                + ", ".join(self.unenforced_axes)
                + " -- anything on these axes is leaked, results are an upper bound"
            )
        if self.leaking_axes:
            lines.append("  LEAKING: " + ", ".join(self.leaking_axes))
        else:
            lines.append("  no enforced axis leaks")
        return "\n".join(lines)


def units_from_labels(labels: Iterable[object], dataset: str | None = None) -> list[TextUnit]:
    """Convert row labels into leakage-checkable units."""
    return [
        TextUnit(
            row_id=label.row_id,
            text=label.text,
            paragraph_id=label.paragraph_id,
            article_id=label.article_id,
            dataset=dataset,
        )
        for label in labels
    ]


def group_ids(units: Iterable[TextUnit], mode: str) -> set[str]:
    """Distinct text groups in ``units`` for a mode."""
    return {unit.group_for(mode) for unit in units}


def check_text_axis(parts: Mapping[str, Sequence[TextUnit]], mode: str) -> AxisReport:
    """Check the text axis: no group may span two parts."""
    report = AxisReport(name="text", mode=mode, enforced=mode != "none")
    report.detail["rows"] = sum(len(v) for v in parts.values())
    if not report.enforced:
        return report

    owners: dict[str, str] = {}
    for part, units in parts.items():
        for unit in units:
            group = unit.group_for(mode)
            if group in owners and owners[group] != part:
                report.violations.append(
                    f"{mode} {group!r} appears in {owners[group]!r} and {part!r}"
                )
            else:
                owners.setdefault(group, part)
    report.detail["groups"] = len(owners)
    return report


def check_subject_axis(parts: Mapping[str, Sequence[str]], mode: str) -> AxisReport:
    """Check the subject axis.

    ``cross_subject`` requires every participant in exactly one part.
    ``one_subject`` requires that only ONE participant appears anywhere, which is
    the deliberately-leaky within-participant setting.
    """
    report = AxisReport(name="subject", mode=mode, enforced=mode != "none")
    subjects = {s for entries in parts.values() for s in entries}
    report.detail["subjects"] = len(subjects)
    if not report.enforced:
        return report

    if mode == "cross_subject":
        seen: dict[str, str] = {}
        for part, entries in parts.items():
            for subject in entries:
                if subject in seen and seen[subject] != part:
                    report.violations.append(
                        f"subject {subject!r} appears in {seen[subject]!r} and {part!r}"
                    )
                else:
                    seen.setdefault(subject, part)
    elif mode == "one_subject":
        if len(subjects) != 1:
            report.violations.append(
                f"{len(subjects)} subjects present but one_subject requires exactly "
                f"1: {sorted(subjects)}"
            )
    else:
        raise LeakageError(f"unknown subject mode {mode!r}; expected one of {SUBJECT_MODES}")
    return report


def check_sex_axis(
    parts: Mapping[str, Sequence[str]],
    participants: Mapping[str, Participant],
) -> AxisReport:
    """Check sex balance, treating any imbalance inside a part as leakage.

    Subjects with no recorded sex are reported separately rather than assumed
    balanced, since "unknown" cannot be shown to be even.
    """
    report = AxisReport(name="sex", mode="balanced", enforced=True)
    for part, entries in parts.items():
        counts: Counter[str] = Counter()
        for subject in entries:
            record = participants.get(subject)
            counts[(record.sex if record and record.sex else "unknown")] += 1
        report.detail[part] = dict(counts)
        imbalance = abs(counts.get("F", 0) - counts.get("M", 0))
        if imbalance:
            report.violations.append(
                f"{part!r} sex imbalance |F-M|={imbalance} "
                f"(F={counts.get('F', 0)}, M={counts.get('M', 0)})"
            )
        if counts.get("unknown"):
            report.violations.append(
                f"{part!r} has {counts['unknown']} subject(s) with no recorded sex"
            )
    return report


def check_run_axis(parts: Mapping[str, Sequence[str]]) -> AxisReport:
    """Check that no recording appears in two parts."""
    report = AxisReport(name="run", mode="disjoint", enforced=True)
    owners: dict[str, str] = {}
    for part, runs in parts.items():
        for run in runs:
            if run in owners and owners[run] != part:
                report.violations.append(
                    f"recording {run!r} appears in {owners[run]!r} and {part!r}"
                )
            else:
                owners.setdefault(run, part)
    report.detail["runs"] = len(owners)
    return report


def evaluate_matrix(
    *,
    subject_parts: Mapping[str, Sequence[str]],
    text_parts: Mapping[str, Sequence[TextUnit]] | None = None,
    participants: Mapping[str, Participant] | None = None,
    run_parts: Mapping[str, Sequence[str]] | None = None,
    subject_mode: str = DEFAULT_SUBJECT_MODE,
    text_mode: str = DEFAULT_LEAKAGE_MODE,
    check_sex: bool = True,
) -> LeakageMatrix:
    """Evaluate every axis and return the full matrix.

    Always returns a report, even when nothing is enforced, so callers can print
    it unconditionally and silence never has to be interpreted as safety.

    ``check_sex`` is False for single-subject splits, where there is nothing to
    balance and reporting an imbalance would be noise rather than signal.
    """
    return LeakageMatrix(
        subject=check_subject_axis(subject_parts, subject_mode),
        text=check_text_axis(text_parts or {}, text_mode),
        sex=(
            check_sex_axis(subject_parts, participants or {})
            if check_sex
            else AxisReport(name="sex", mode="n/a", enforced=False)
        ),
        runs=check_run_axis(run_parts or {}),
    )


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
    requested ratios while guaranteeing that a group is never divided -- half a
    paragraph in train and half in test is exactly the leak to avoid.
    """
    from collections import defaultdict

    if mode not in TEXT_MODES:
        raise LeakageError(f"unknown text mode {mode!r}; expected one of {TEXT_MODES}")

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
    order = sorted(grouped.items(), key=lambda kv: (-len(kv[1]), rng.random()))

    assigned: dict[str, list[TextUnit]] = {part: [] for part in ratios}
    for _, group_units in order:
        part = max(
            ratios,
            key=lambda p: (targets[p] - len(assigned[p]), -list(ratios).index(p)),
        )
        assigned[part].extend(group_units)

    return {part: tuple(sorted(group, key=lambda u: u.row_id)) for part, group in assigned.items()}


def summarize_coverage(units: Sequence[TextUnit], mode: str) -> str:
    """Human-readable text-structure coverage, so weak structure is visible."""
    if not units:
        return "no text units"
    total = len(units)
    n_par = sum(1 for u in units if u.paragraph_id)
    n_art = sum(1 for u in units if u.article_id)
    groups = len(group_ids(units, mode)) if mode != "none" else total
    return (
        f"{total} rows; {n_par} with a paragraph ({n_par / total:.1%}), "
        f"{n_art} with an article ({n_art / total:.1%}); "
        f"{groups} {mode}-level groups"
    )


def supports_text_mode(dataset: str, mode: str) -> bool:
    """Whether a dataset can support a text mode.

    Chisco stimuli are standalone sentences with no surrounding document, so
    paragraph and article identity cannot be recovered there; grouping by those
    modes would be vacuous rather than strict.
    """
    if mode in {"none", "row"}:
        return True
    return dataset in DOCUMENT_DATASETS


def pick_one_subject(
    participants: Sequence[Participant],
    *,
    text_mode: str = DEFAULT_LEAKAGE_MODE,
    seed: int = 0,
) -> Participant:
    """Choose a single participant at random for a within-subject split.

    Ignores sex balance deliberately: with one participant there is nothing to
    balance, and the point of this mode is a within-participant experiment.
    When the text mode needs document structure, only datasets shipping source
    documents are eligible, since the constraint could not be evaluated
    otherwise.
    """
    import random

    eligible = [p for p in participants if supports_text_mode(p.dataset or "", text_mode)]
    if not eligible:
        raise LeakageError(
            f"no participant can support text mode {text_mode!r}: it needs a dataset "
            f"with source documents ({', '.join(DOCUMENT_DATASETS)})"
        )
    return random.Random(seed).choice(sorted(eligible, key=lambda p: p.subject_id))


def describe_subject_mode(mode: str) -> str:
    """One-line explanation of a subject mode."""
    return {
        "none": "participants may appear in any part (leaky; upper bound)",
        "one_subject": "all parts from a single participant (leaky; no generalisation)",
        "cross_subject": "no participant appears in two parts (default)",
    }.get(mode, f"unknown subject mode {mode!r}")


def describe_text_mode(mode: str) -> str:
    """One-line explanation of a text mode."""
    return {
        "none": "text may repeat across parts (leaky; upper bound)",
        "row": "no identical row in two parts (default)",
        "paragraph": "no paragraph in two parts",
        "article": "no chapter in two parts (strictest)",
    }.get(mode, f"unknown text mode {mode!r}")


def default_text_mode() -> str:
    return DEFAULT_LEAKAGE_MODE


def default_subject_mode() -> str:
    return DEFAULT_SUBJECT_MODE
