"""Building training partitions directly from a materialised split tree.

A split is a directory of symlinks::

    splits/train/sub-04 -> ../../bci-eeg/ds005170/sub-04
    splits/test/sub-01  -> ../../bci-eeg/ds005170/sub-01

Training therefore needs no separate cache: the tree *is* the partition. This
module walks a part, resolves the links, and yields the recordings belonging to
it. That is what makes subject-level and run-level splits behave identically
from the trainer's point of view -- the only difference is what the links point
at, which is the point of materialising them as links rather than a manifest.

Two things are deliberately strict:

* **No trial may appear in two parts.** A recording that resolves to the same
  file under both ``train`` and ``test`` is participant leakage, which inflates
  scores while looking like a good model. It is detected, not tolerated.
* **The vocabulary is built from training text only.** Fitting the tokenizer on
  the evaluation split leaks its characters into the label space.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

SPLIT_MANIFEST = "split.json"

#: Parts recognised in a split tree, in the order they are reported.
KNOWN_PARTS = ("train", "valid", "test")


class SplitTreeError(RuntimeError):
    """Raised when a split tree is missing, malformed, or leaking."""


@dataclass(frozen=True)
class PartEntry:
    """One entry in a split part: a subject, or a single run of one."""

    part: str
    #: Dataset name when the split spans corpora, else None.
    dataset: str | None
    subject: str
    #: Run identifier for run-level splits, else None.
    run: str | None
    #: Resolved filesystem paths behind this entry.
    paths: tuple[Path, ...] = ()

    @property
    def run_files(self) -> tuple[Path, ...]:
        """Recordings (not sidecars) belonging to this entry."""
        suffixes = {".edf", ".vhdr", ".eeg", ".fif", ".set"}
        return tuple(p for p in self.paths if p.suffix.lower() in suffixes)

    def __str__(self) -> str:
        bits = [self.dataset] if self.dataset else []
        bits.append(self.subject)
        if self.run:
            bits.append(self.run)
        return "/".join(bits)


@dataclass
class SplitTree:
    """A loaded split tree."""

    root: Path
    parts: dict[str, list[PartEntry]] = field(default_factory=dict)
    manifest: dict = field(default_factory=dict)

    @property
    def subject_level(self) -> bool:
        return bool(self.manifest.get("subject_level", True))

    @property
    def warnings(self) -> list[str]:
        return list(self.manifest.get("warnings", []))

    def subjects(self, part: str) -> list[str]:
        """Distinct subject ids in a part, preserving order."""
        seen: dict[str, None] = {}
        for entry in self.parts.get(part, []):
            key = f"{entry.dataset}:{entry.subject}" if entry.dataset else entry.subject
            seen.setdefault(key, None)
        return list(seen)

    def n_files(self, part: str) -> int:
        return sum(len(entry.run_files) for entry in self.parts.get(part, []))


def load_split_tree(path: str | Path) -> SplitTree:
    """Read a materialised split tree.

    Accepts either the split directory itself or a file inside it. Fails loudly
    when the structure is missing rather than silently returning an empty split,
    which would train on nothing and look like a modelling failure.
    """
    path = Path(path)
    if path.is_file():
        path = path.parent
    if not path.is_dir():
        raise SplitTreeError(
            f"no split tree at {path}. Create one first:\n"
            "  bci-eeg-to-text split --dataset <name> --link"
        )

    manifest_path = path / SPLIT_MANIFEST
    manifest: dict = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SplitTreeError(f"{manifest_path} is not valid JSON: {exc}") from exc

    parts: dict[str, list[PartEntry]] = {}
    for part_dir in sorted(p for p in path.iterdir() if p.is_dir()):
        part = part_dir.name
        entries: list[PartEntry] = []
        # Layout is part/[dataset/]<subject>[/<run files>]
        for child in sorted(part_dir.iterdir()):
            if child.is_symlink():
                # <part>/<subject> -> corpus subject directory
                entries.extend(_entries_from_subject(part, None, child))
            elif child.is_dir() and _looks_like_subject(child.name):
                # <part>/<subject>/... : run-level layout for one corpus
                entries.extend(_entries_from_subject(part, None, child))
            elif child.is_dir():
                # <part>/<dataset>/<subject>/... : merged layout
                for subject_dir in sorted(child.iterdir()):
                    entries.extend(_entries_from_subject(part, child.name, subject_dir))
            else:
                entries.append(PartEntry(part, None, child.stem, None, (child.resolve(),)))
        parts[part] = entries

    if not parts:
        raise SplitTreeError(f"{path} contains no parts; is it a split tree?")
    return SplitTree(root=path, parts=parts, manifest=manifest)


def _entries_from_subject(part: str, dataset: str | None, node: Path) -> list[PartEntry]:
    """Turn one subject node (symlink or directory) into entries.

    ``node`` is always a subject. What varies is how its files are arranged:

    * a symlink to a corpus subject directory (subject-level split), or
    * a real directory of run subdirectories, each possibly nested under a
      session because run numbers repeat across sessions
      (``<subject>/<session>/<run>/<files>``).
    """
    if node.is_symlink():
        target = node.resolve()
        if target.is_dir():
            return [PartEntry(part, dataset, node.name, None, _files_under(target))]
        return [PartEntry(part, dataset, node.stem, None, (target,))]

    # Real directory: group files by run, keeping the node as the subject.
    # Walk to the leaves rather than assuming a fixed depth, since a run may sit
    # directly under the subject or under a session directory.
    by_run: dict[str, list[Path]] = {}
    for path in sorted(node.rglob("*")):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if not resolved.exists():
            continue
        by_run.setdefault(_run_of(resolved.name) or "run", []).append(resolved)
    return [
        PartEntry(part, dataset, node.name, run, tuple(paths))
        for run, paths in sorted(by_run.items())
    ]


def _looks_like_subject(name: str) -> bool:
    """True when a directory name is a BIDS subject id rather than a dataset.

    The two layouts are `<part>/<subject>/...` and `<part>/<dataset>/<subject>`,
    so distinguishing them by name shape avoids guessing from depth alone.
    """
    return name.startswith("sub-")


def _files_under(directory: Path) -> tuple[Path, ...]:
    """Recordings beneath a subject directory, following the tree."""
    out: list[Path] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".edf", ".vhdr", ".eeg", ".fif", ".set"}:
            out.append(path)
    return tuple(out)


def _run_of(filename: str) -> str | None:
    """Extract a BIDS ``run-NN`` token from a filename, if present."""
    for token in filename.replace(".", "_").split("_"):
        if token.startswith("run-"):
            return token
    return None


def check_no_leakage(tree: SplitTree) -> None:
    """Assert that no recording appears in more than one part.

    Identity is the resolved path, so a subject reachable through two different
    links is still caught. This is the failure mode that makes a model look good
    while generalising badly, so it is an error rather than a warning.
    """
    owners: dict[Path, str] = {}
    for part, entries in tree.parts.items():
        for entry in entries:
            for path in entry.run_files:
                resolved = path.resolve()
                if resolved in owners and owners[resolved] != part:
                    raise SplitTreeError(
                        f"{resolved.name} appears in both {owners[resolved]!r} and "
                        f"{part!r}; this leaks data across parts and inflates "
                        "scores. Regenerate the split."
                    )
                owners[resolved] = part


def iter_part_files(tree: SplitTree, part: str) -> Iterator[Path]:
    """Yield resolved recordings belonging to ``part``, sorted."""
    for entry in tree.parts.get(part, []):
        yield from entry.run_files


def summarise(tree: SplitTree) -> str:
    """One-line description of a split tree for logging."""
    bits = []
    for part in tree.parts:
        bits.append(f"{part}: {len(tree.subjects(part))} subjects, {tree.n_files(part)} files")
    level = "subject" if tree.subject_level else "run"
    return f"{level}-level split at {tree.root} ({'; '.join(bits)})"
