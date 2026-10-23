"""Smoke-test selection: a tiny, honest slice of a corpus.

A benchmark that runs on a toy slice must still respect the properties that make
a number meaningful, otherwise the smoke test measures nothing:

* **one subject** -- trials from several participants would mix signal
  characteristics that the real split holds out;
* **two runs per text** -- so a text can land in train in one fold and in test in
  another, which is what a cross-validated text benchmark needs;
* **five texts, ten runs** -- small enough to finish in minutes;
* **5-fold cross-validation, 3-1-1** -- three folds train, one validates, one
  tests, and every run is tested exactly once.

Failures raise :class:`SmokeSelectionError` rather than returning a smaller
sample: silently shrinking the smoke set is how a benchmark ends up comparing
numbers computed on different data.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The text used by a run, as recorded in the streaming index.
RowLike = dict


class SmokeSelectionError(RuntimeError):
    """A dataset cannot supply the requested smoke sample."""


@dataclass(frozen=True)
class SmokeRun:
    """One run and the distinct texts it contains."""

    run_id: str
    subject: str
    run: str
    texts: tuple[str, ...]


@dataclass(frozen=True)
class SmokeSelection:
    """The chosen slice: its subject, texts, runs and their row indices."""

    subject: str
    texts: tuple[str, ...]
    runs: tuple[SmokeRun, ...]
    row_indices: tuple[int, ...]

    @property
    def run_ids(self) -> tuple[str, ...]:
        return tuple(r.run_id for r in self.runs)


def group_runs_by_text(rows: list[RowLike], *, min_runs: int = 1) -> dict[str, list[str]]:
    """Map each distinct text to the run ids containing it.

    Texts found in fewer than ``min_runs`` runs are dropped, since they cannot
    support the train/test pairing the smoke test needs.
    """
    index: dict[str, dict[str, None]] = {}
    for row in rows:
        text = str(row["text"])
        # Skip empty and single-character banner rows: a chapter marker like "1"
        # matches every run trivially and would dominate the selection.
        if len(text.strip()) < 2:
            continue
        index.setdefault(text, {})[str(row["run_id"])] = None
    return {text: list(runs) for text, runs in index.items() if len(runs) >= min_runs}


def select_smoke_runs(
    rows: list[RowLike],
    *,
    seed: int = 0,
    n_texts: int = 5,
    runs_per_text: int = 2,
) -> SmokeSelection:
    """Pick ``n_texts`` texts and ``runs_per_text`` runs for each, from one subject."""
    by_subject: dict[str, list[RowLike]] = {}
    for row in rows:
        by_subject.setdefault(str(row["subject"]), []).append(row)

    import random

    rng = random.Random(seed)
    subjects = sorted(by_subject)
    rng.shuffle(subjects)
    reason = "no subject has enough runs"
    for subject in subjects:
        subject_rows = by_subject[subject]
        groups = group_runs_by_text(subject_rows, min_runs=runs_per_text)
        if len(groups) < n_texts:
            reason = (
                f"subject {subject} has {len(groups)} texts shared by at least "
                f"{runs_per_text} runs, need {n_texts}"
            )
            continue
        runs_of: dict[str, list[RowLike]] = {}
        for row in subject_rows:
            runs_of.setdefault(str(row["run_id"]), []).append(row)
        if len(runs_of) < n_texts * runs_per_text:
            reason = f"subject {subject} has {len(runs_of)} runs, need {n_texts * runs_per_text}"
            continue

        texts = sorted(groups)
        rng.shuffle(texts)
        chosen_texts: list[str] = []
        chosen_runs: set[str] = set()
        for text in texts:
            if len(chosen_texts) == n_texts:
                break
            candidates = [r for r in groups[text] if r not in chosen_runs]
            if len(candidates) < runs_per_text:
                continue
            rng.shuffle(candidates)
            picked = candidates[:runs_per_text]
            chosen_texts.append(text)
            chosen_runs.update(picked)
        if len(chosen_texts) < n_texts or len(chosen_runs) < n_texts * runs_per_text:
            reason = f"subject {subject}: texts overlap too heavily to pair disjointly"
            continue

        ordered_runs = sorted(chosen_runs)
        runs = []
        for run_id in ordered_runs:
            texts_in_run = tuple(
                sorted(
                    {str(r["text"]) for r in runs_of[run_id] if len(str(r["text"]).strip()) >= 2}
                )
            )
            sample = runs_of[run_id][0]
            runs.append(
                SmokeRun(
                    run_id=run_id,
                    subject=subject,
                    run=str(sample["run"]),
                    texts=texts_in_run,
                )
            )
        row_indices = tuple(i for i, row in enumerate(rows) if str(row["run_id"]) in chosen_runs)
        return SmokeSelection(
            subject=subject,
            texts=tuple(chosen_texts),
            runs=tuple(runs),
            row_indices=row_indices,
        )

    raise SmokeSelectionError(reason)


def fold_indices(n_items: int, *, n_folds: int = 5) -> list[tuple[list[int], list[int], list[int]]]:
    """Split ``range(n_items)`` into folds of 3-1-1 (train/valid/test).

    With 10 items and 5 folds each fold is 6 train / 2 valid / 2 test, and each
    item appears in the test slice exactly once.
    """
    if n_items % n_folds != 0:
        raise ValueError(
            f"{n_items} items are not divisible into {n_folds} folds; "
            "the smoke sample must be n_folds * group_size"
        )
    size = n_items // n_folds
    folds = []
    for f in range(n_folds):
        test = list(range(f * size, (f + 1) * size))
        valid = list(range(((f + 1) % n_folds) * size, ((f + 1) % n_folds + 1) * size))
        train = [i for i in range(n_items) if i not in set(test) | set(valid)]
        folds.append((train, valid, test))
    return folds
