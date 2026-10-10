"""Text structure for leakage control: rows, paragraphs and articles.

A model can exploit more than a shared recording. Two splits can hold entirely
different EEG while sharing text, and the model can still memorise
sentence-to-sentence regularities rather than learn to read. This module
recovers the textual hierarchy so splits can be checked at whatever granularity
the user wants.

Two facts about this corpus make that possible, both verified:

* The original novels are plain text with ``ChN`` markers delimiting chapters
  (articles) and blank lines delimiting paragraphs.
* **Rows are verbatim substrings of the original.** Measured over the first six
  runs of each novel: 2064/2069 LittlePrince rows and 3077/3077 GarnettDream
  rows appear exactly in the source text. So a row's position can be located by
  string search, and its paragraph and chapter follow from that offset.

Granularities, from weakest to strongest:

======================  ==================================================
``none``                nothing is enforced
``row``                 no identical row text in two parts
``paragraph``           no paragraph appears in two parts
``article``             no chapter appears in two parts (strictest)
======================  ==================================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from bci_eeg_to_text.tokenizer import normalize_text

#: Leakage modes, weakest to strongest.
LEAKAGE_MODES = ("none", "row", "paragraph", "article")

#: Default: rows never repeat across parts, but prose may.
DEFAULT_LEAKAGE_MODE = "row"

#: Chapter markers in the source novels, e.g. "Ch0", " Ch12".
_CHAPTER_RE = re.compile(r"^\s*Ch(\d+)\s*$")

#: Novels and the encoding each is stored in. The two differ, which is why
#: encoding is explicit rather than assumed.
NOVEL_ENCODINGS = {"LittlePrince": "utf-8", "GarnettDream": "gbk"}

#: Text formats that can delimit paragraphs and articles.
PLAIN_TEXT_SUFFIXES = (".txt", ".md")


class TextStructureError(RuntimeError):
    """Raised when text structure cannot be recovered."""


@dataclass
class Document:
    """A source document split into ordered paragraphs and chapters."""

    name: str
    #: Concatenation of every non-whitespace character, for offset lookup.
    flat: str = ""
    #: Half-open [start, stop) char ranges into ``flat``, one per paragraph.
    paragraphs: list[tuple[int, int]] = field(default_factory=list)
    #: Article (chapter) id per paragraph, or None when unknown.
    paragraph_article: list[str | None] = field(default_factory=list)
    #: Article (chapter) id per character range.
    articles: list[tuple[str, int, int]] = field(default_factory=list)
    #: Per-needle search cursor, so repeated rows resolve to successive
    #: occurrences rather than always the first.
    _search_from: dict[str, int] = field(default_factory=dict, repr=False)

    def locate(self, text: str) -> int | None:
        """Character offset of ``text`` in the document, or None if absent.

        A per-needle cursor makes a repeated row resolve to the *next* unused
        occurrence rather than always the first, which would collapse distinct
        occurrences onto one paragraph. Use :meth:`reset_cursor` (or a fresh
        Document) when starting a new sequence of rows, because a document read
        contains rows from the beginning again.
        """
        needle = normalize_text(text)
        if not needle:
            return None
        start = self._search_from.get(needle, 0)
        found = self.flat.find(needle, start)
        if found < 0:
            # Wrap around: the cursor may have advanced past the only
            # occurrence, which happens when a new file restarts the document.
            found = self.flat.find(needle)
            if found < 0:
                return None
        self._search_from[needle] = found + len(needle)
        return found

    def reset_cursor(self) -> None:
        """Forget search history so scanning restarts at the document start."""
        self._search_from.clear()

    def paragraph_of(self, offset: int) -> int | None:
        """Index of the paragraph containing a character offset."""
        for index, (start, stop) in enumerate(self.paragraphs):
            if start <= offset < stop:
                return index
        return None

    def article_of(self, offset: int) -> str | None:
        """Article id containing a character offset."""
        for name, start, stop in self.articles:
            if start <= offset < stop:
                return name
        return None

    def paragraph_id(self, offset: int) -> str | None:
        """Stable identifier for the paragraph at ``offset``."""
        index = self.paragraph_of(offset)
        return f"{self.name}#p{index}" if index is not None else None


def _looks_like_prose(line: str) -> bool:
    """True for likely novel text rather than a chapter heading or blank."""
    stripped = line.strip()
    if not stripped:
        return False
    return not _CHAPTER_RE.match(stripped)


def parse_document(text: str, name: str) -> Document:
    """Split raw novel text into paragraphs and articles.

    Paragraphs are runs of consecutive non-blank lines; a blank line ends one.
    ``ChN`` markers start a new article and are recorded as boundaries rather
    than as prose.
    """
    document = Document(name=name)
    normalized_lines: list[str] = []
    chapter_starts: list[tuple[str, int]] = []

    for line in text.splitlines():
        match = _CHAPTER_RE.match(line)
        if match:
            chapter_starts.append((f"Ch{int(match.group(1))}", len(normalized_lines)))
            normalized_lines.append("")  # keep offsets aligned with markers
            continue
        normalized_lines.append(normalize_text(line) if _looks_like_prose(line) else "")

    document.flat = "".join(normalized_lines)

    # Paragraphs: consecutive non-empty lines.
    cursor = 0
    current: list[int] = []
    boundaries: list[tuple[int, int]] = []
    for index, line in enumerate(normalized_lines):
        if line:
            if not current:
                current = [index, index]
            else:
                current[1] = index
        elif current:
            boundaries.append((current[0], current[1]))
            current = []
    if current:
        boundaries.append((current[0], current[1]))

    line_offsets = []
    for line in normalized_lines:
        line_offsets.append(cursor)
        cursor += len(line)

    for start_line, stop_line in boundaries:
        start = line_offsets[start_line]
        stop = line_offsets[stop_line] + len(normalized_lines[stop_line])
        if stop > start:
            document.paragraphs.append((start, stop))

    # Articles span from a chapter marker to the next one.
    chapter_offsets = [(name_, line_offsets[line]) for name_, line in chapter_starts]
    for order, (chapter, start) in enumerate(chapter_offsets):
        stop = (
            chapter_offsets[order + 1][1]
            if order + 1 < len(chapter_offsets)
            else len(document.flat)
        )
        document.articles.append((chapter, start, stop))

    document.paragraph_article = [document.article_of(start) for start, _ in document.paragraphs]
    return document


def read_document(path: str | Path, name: str | None = None) -> Document:
    """Read a source document, decoding with the encoding this corpus uses."""
    path = Path(path)
    if not path.exists():
        raise TextStructureError(f"no such text file: {path}")
    name = name or path.stem
    encoding = NOVEL_ENCODINGS.get(name, "utf-8")
    try:
        text = path.read_text(encoding=encoding)
    except UnicodeDecodeError:
        # Fall back rather than failing: the structure still parses.
        text = path.read_bytes().decode(encoding, errors="replace")
    return parse_document(text, name)


def find_source_documents(root: str | Path) -> dict[str, Path]:
    """Locate plain-text source novels beneath a dataset root.

    Returns a mapping of document name to path, which is what lets rows be
    traced back to paragraphs and chapters.
    """
    from bci_eeg_to_text.eeg import is_annex_pointer

    root = Path(root)
    found: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in PLAIN_TEXT_SUFFIXES or not path.is_file():
            continue
        # Skip annex pointers (named by hash) and editor/checkpoint copies:
        # neither is the authoritative novel.
        if is_annex_pointer(path):
            continue
        if "checkpoint" in path.stem.lower():
            continue
        if not path.stem.isascii() or len(path.stem) > 40:
            continue
        found.setdefault(path.stem, path)
    return found


@dataclass
class RowLabel:
    """Textual identity of one supervised row."""

    text: str
    row_id: str
    paragraph_id: str | None = None
    article_id: str | None = None
    document: str | None = None


def label_rows(
    texts: list[str],
    document: Document,
) -> list[RowLabel]:
    """Attach row, paragraph and article identity to each text row.

    A row whose text cannot be located in the source document keeps its row id
    but gets no paragraph or article, so it can never match a stricter check by
    accident. Callers can count these to see how complete the mapping is.
    """
    labels: list[RowLabel] = []
    for index, text in enumerate(texts):
        row_id = f"{document.name}#r{index}"
        offset = document.locate(text)
        if offset is None:
            labels.append(RowLabel(text=text, row_id=row_id, document=document.name))
            continue
        labels.append(
            RowLabel(
                text=text,
                row_id=row_id,
                paragraph_id=document.paragraph_id(offset),
                article_id=document.article_of(offset),
                document=document.name,
            )
        )
    return labels
