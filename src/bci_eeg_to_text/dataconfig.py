"""Dataset location resolution and symlinked split materialisation.

Two ideas drive this module:

**Data location is configuration, not a hard-coded path.** The dataset root and
the subdirectory holding the corpora are both configurable, because neither is
under our control: the drive may be remounted, and a user may keep the corpora
somewhere other than ``bci-eeg``.

**A split should be visible, not encoded.** Copying recordings to represent a
split is impossible here (419 GB of data against ~15 GB free) and a JSON
manifest alone is opaque, since you cannot see which files belong to a split by
looking at the filesystem. Materialising a split as a **tree of relative
symlinks** gives a layout you can browse with ``ls``, costs no disk space, and
keeps the recordings exactly where they are.

The symlink targets are always *relative*, so the tree keeps working when the
whole directory (dataset included) is moved or remounted elsewhere.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from bci_eeg_to_text.splits import Split

#: Environment variable naming the root under which datasets live.
DATA_ROOT_ENV = "EEG_TEXT_BCI_DATA_ROOT"

#: Environment variable naming the subdirectory holding the corpora.
DATA_SUBDIR_ENV = "EEG_TEXT_BCI_DATA_SUBDIR"

#: Environment variable allowing writes inside the dataset root.
DATA_WRITABLE_ENV = "EEG_TEXT_BCI_DATA_WRITABLE"

#: Default location, used only when the environment provides nothing.
DEFAULT_DATA_ROOT = "eeg-text-bci-data"

#: Default corpus subdirectory under the root.
DEFAULT_DATA_SUBDIR = "bci-eeg"

#: Directory name for a materialised split inside the dataset root.
SPLIT_DIRNAME = "splits"


class DataAccessError(RuntimeError):
    """Raised when the configured dataset location cannot be used as assumed."""


def _env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment flag, accepting common spellings."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class DataConfig:
    """Where the data lives and how it may be treated.

    ``assume_writable`` defaults to False because the corpora are typically a
    read-only mount. When False, splits are written to the current working
    directory instead of inside the dataset root, and nothing under the root is
    ever modified.
    """

    #: Directory containing the corpus subdirectories.
    root: Path = Path(DEFAULT_DATA_ROOT)
    #: Subdirectory of ``root`` holding the corpora (never hard-coded elsewhere).
    subdir: str = DEFAULT_DATA_SUBDIR
    #: Whether splits may be written inside ``root``.
    assume_writable: bool = False

    @classmethod
    def from_env(
        cls,
        *,
        root: str | Path | None = None,
        subdir: str | None = None,
        assume_writable: bool | None = None,
    ) -> DataConfig:
        """Build a config from explicit values, falling back to the environment.

        Precedence is explicit argument, then environment, then default, so the
        CLI can override the environment without the environment being ignored.
        """
        resolved_root = root if root is not None else os.environ.get(DATA_ROOT_ENV)
        resolved_subdir = subdir if subdir is not None else os.environ.get(DATA_SUBDIR_ENV)
        resolved_writable = (
            assume_writable
            if assume_writable is not None
            else _env_flag(DATA_WRITABLE_ENV, default=False)
        )
        return cls(
            root=Path(resolved_root or DEFAULT_DATA_ROOT),
            subdir=resolved_subdir or DEFAULT_DATA_SUBDIR,
            assume_writable=bool(resolved_writable),
        )

    @property
    def corpora_dir(self) -> Path:
        """Directory holding the individual corpora (e.g. ``ds005170``)."""
        return self.root / self.subdir

    def dataset_path(self, name: str) -> Path:
        """Path to one corpus by its OpenNeuro accession directory name."""
        return self.corpora_dir / name

    def split_root(self, *, cwd: Path | None = None) -> Path:
        """Where materialised splits should be written.

        Inside the dataset root when it is declared writable, otherwise a
        ``splits/`` directory in the current working directory. This is the
        whole point of ``assume_writable``: never write to a read-only mount by
        accident.
        """
        if self.assume_writable:
            return self.corpora_dir / SPLIT_DIRNAME
        return (cwd or Path.cwd()) / SPLIT_DIRNAME

    def is_inside_root(self, path: Path) -> bool:
        """True if ``path`` resolves within the dataset root."""
        try:
            path.resolve().relative_to(self.root.resolve())
            return True
        except (ValueError, OSError):
            return False


@dataclass
class MaterialisedSplit:
    """Result of writing a split tree."""

    path: Path
    n_links: int
    parts: dict[str, int] = field(default_factory=dict)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    #: Small text sidecars copied in so the pairing is human-readable.
    n_text_copies: int = 0

    @property
    def manifest_path(self) -> Path:
        return self.path / "split.json"


def relative_symlink(target: Path, link: Path) -> None:
    """Create ``link`` pointing at ``target`` using a relative path.

    Relative targets keep the tree valid when the dataset and the split are
    moved or remounted together, which an absolute link would not survive.
    """
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    # Resolve both sides so the relative path is correct even if one is itself
    # a symlink (the dataset mount usually is).
    relative = os.path.relpath(target.resolve(), link.parent.resolve())
    link.symlink_to(relative)


def materialise_split(
    split: Split,
    *,
    data: DataConfig,
    corpus_paths: dict[str, Path],
    cwd: Path | None = None,
    link_runs: bool = True,
    leakage_mode: str | None = None,
    leakage_violations: int | None = None,
    copy_text: bool = True,
) -> MaterialisedSplit:
    """Write a split as a tree of relative symlinks plus a manifest.

    Layout::

        splits/<split_name>/<part>/<subject>[@dataset]/<run files>

    For run-level splits each entry is ``subject/run`` and individual run files
    are linked; for subject-level splits the whole per-subject directory is
    linked, which is fewer links and easier to browse.

    ``corpus_paths`` maps a dataset name to its directory, so the same function
    serves single-corpus and merged splits.
    """
    root = data.split_root(cwd=cwd)
    # `assume_writable` is a user assertion; when it is wrong the failure should
    # name the flag and the fix rather than surfacing a bare OSError.
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DataAccessError(
            f"cannot write the split to {root}: {exc}. "
            f"{data.corpora_dir} appears to be read-only. Drop --data-writable / "
            f"{DATA_WRITABLE_ENV} to write the split to the current directory "
            "instead (symlinks are relative, so either location works)."
        ) from exc
    result = MaterialisedSplit(path=root, n_links=0)

    for part, entries in split.parts.items():
        for entry in entries:
            if "/" in entry:
                subject, run = entry.split("/", 1)
            else:
                subject, run = entry, None

            dataset = None
            if ":" in subject:
                dataset, subject = subject.split(":", 1)
            if dataset is None:
                dataset = next(iter(corpus_paths))

            corpus = corpus_paths.get(dataset)
            if corpus is None or not corpus.exists():
                result.skipped.append((entry, f"corpus {dataset!r} not available"))
                continue

            link_root = root / part
            if dataset and len(corpus_paths) > 1:
                link_root = link_root / dataset

            if run is None:
                source = _subject_dir(corpus, subject)
                if source is None:
                    result.skipped.append((entry, "no directory for subject"))
                    continue
                link = link_root / source.name
                relative_symlink(source, link)
                result.n_links += 1
                if copy_text:
                    # Write text beside the link, never through it: the link
                    # points into the (read-only) dataset, and a copy landing
                    # there would be both an error and a surprise.
                    text_dir = link_root / f"{source.name}.text"
                    result.n_text_copies += copy_text_companions(text_dir, corpus, subject)
            else:
                sources = _run_files(corpus, subject, run)
                if not sources:
                    result.skipped.append((entry, "no files for run"))
                    continue
                # Nest by subject AND run. Flattening all of a subject's runs
                # into one directory makes two different parts able to write
                # the same path -- which silently leaks a run across train and
                # test when the two corpora share a subject id.
                # `run` may be "ses-X/run-01"; nesting by it reproduces the
                # session layer so two same-numbered runs never collide.
                run_dir = link_root / subject / Path(run)
                session, _, bare = run.rpartition("/")
                for source in sources:
                    relative_symlink(source, run_dir / source.name)
                    result.n_links += 1
                if copy_text:
                    result.n_text_copies += copy_text_companions(
                        run_dir, corpus, subject, session=session or None, run=bare
                    )
                _ = (session, bare)

            result.parts[part] = result.parts.get(part, 0) + 1

    manifest = {
        "level": "run" if not split.subject_level else "subject",
        "leakage_mode": leakage_mode,
        "leakage_violations": leakage_violations,
        "subject_level": split.subject_level,
        "warnings": list(split.warnings),
        "data_root": str(data.root),
        "data_subdir": data.subdir,
        "writable_root": data.assume_writable,
        "parts": {k: list(v) for k, v in split.parts.items()},
        "n_links": result.n_links,
        "n_text_copies": result.n_text_copies,
        "skipped": result.skipped,
    }
    result.manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def copy_text_companions(
    entry_dir: Path,
    corpus: Path,
    subject: str,
    *,
    session: str | None = None,
    run: str | None = None,
) -> int:
    """Copy small text sidecars next to a subject's links.

    Recordings stay symlinked because they are gigabytes. The text files that
    say *what the participant was reading* are kilobytes, so copying them is
    cheap and makes the pairing inspectable: `ls` inside a split shows the EEG
    links and the stimulus text side by side, rather than leaving a reader to
    re-derive which row applies to which recording.

    Only files that are plainly text are copied; data formats are never touched.
    """
    import shutil

    # Everything is filtered by size. The point is to copy the small text that
    # describes what was read, never the recordings.
    max_bytes = 2 * 1024 * 1024
    candidates: list[Path] = []

    # Per-subject text (events, channels, stimulus lists).
    for pattern in ("*.txt", "*.tsv", "*.json", "*.csv"):
        candidates.extend(corpus.glob(f"**/{subject}_*{pattern}"))
        candidates.extend(corpus.glob(f"**/{subject}/*{pattern}"))

    # Dataset-level text: the participants table and stimulus maps describe the
    # pairing even though they are not subject-prefixed.
    for pattern in ("participants.tsv", "*.json", "*.txt", "*.csv"):
        candidates.extend(corpus.glob(pattern))
        text_dir = corpus / "json"
        if text_dir.is_dir():
            candidates.extend(text_dir.glob(pattern))

    copied = 0
    for source in sorted(set(candidates)):
        if not source.is_file() or source.is_symlink():
            continue
        try:
            if source.stat().st_size > max_bytes:
                continue
        except OSError:
            continue
        # Session/run filters apply only to per-subject files. Dataset-level
        # text such as participants.tsv describes every part and is kept as is.
        is_subject_file = source.name.startswith(subject)
        fields = source.name.replace(".", "_").split("_")
        if is_subject_file and session and session not in fields:
            continue
        if is_subject_file and run:
            wanted = _run_number(run)
            file_runs = {_run_number(f) for f in fields if f.startswith("run-")}
            if file_runs and wanted not in file_runs:
                continue
        target = entry_dir / source.name
        if target.exists():
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied += 1
        except OSError:
            continue
    return copied


def _subject_dir(corpus: Path, subject: str) -> Path | None:
    """Locate a subject's directory within a corpus."""
    direct = corpus / subject
    if direct.is_dir():
        return direct
    # Some corpora nest subjects under a session or derivative layer.
    for candidate in sorted(corpus.glob(f"*/{subject}")):
        if candidate.is_dir():
            return candidate
    matches = sorted(corpus.glob(f"**/{subject}"))
    return matches[0] if matches else None


def _run_files(corpus: Path, subject: str, run: str) -> list[Path]:
    """Every usable data file belonging to one run of one subject.

    Matches the BIDS stem (``..._run-01_...``) plus its sidecars so a linked run
    is complete. Two kinds of file are skipped:

    * already-symlinked files, which would produce chains that break when
      relative targets are re-anchored, and
    * git-annex pointer stubs, which are not data at all. The ``derivatives``
      trees are full of 104-byte placeholders for recordings that were never
      fetched, and linking those would suggest data that is not there.
    """
    from bci_eeg_to_text.eeg import is_annex_pointer

    # `run` may be session-qualified ("ses-GarnettDream/run-01") because run
    # numbers repeat across sessions. Requiring the session when given is what
    # stops run-01 of one session from being linked into a split part that only
    # asked for run-01 of the other.
    session, _, bare = run.rpartition("/")
    # Compare run numbers numerically: corpora disagree on zero padding
    # (chisco calls it "run-001" where the file says "run-01"), so a string
    # comparison silently matches nothing.
    wanted = _run_number(bare)
    files: list[Path] = []
    for path in sorted(corpus.glob(f"**/{subject}_*")):
        if not path.is_file() or path.is_symlink():
            continue
        fields = path.name.replace(".", "_").split("_")
        file_runs = {_run_number(f) for f in fields if f.startswith("run-")}
        if wanted is None or wanted not in file_runs:
            continue
        if session and session not in fields:
            continue
        if is_annex_pointer(path):
            continue
        files.append(path)
    return files


def _run_number(token: str) -> int | None:
    """Numeric value of a ``run-NN`` token, ignoring zero padding."""
    _, _, digits = token.partition("run-")
    digits = digits.strip()
    return int(digits) if digits.isdigit() else None


def load_split_manifest(path: str | Path) -> dict:
    """Read a split manifest (or one nested in a split directory)."""
    path = Path(path)
    if path.is_dir():
        path = path / "split.json"
    return json.loads(path.read_text(encoding="utf-8"))
