"""Reading the Chisco stimulus sentences from ``textdataset/*.xlsx``.

The workbook files are read with the standard library only. They are small and
stored as inline strings, so a full spreadsheet dependency is unnecessary and
would add weight to every training environment.

Layout: one header row (``句子`` = sentence, ``标签`` = topic label) followed by
one row per imagined sentence.
"""

from __future__ import annotations

import html
import re
import zipfile
from pathlib import Path

from bci_eeg_to_text.tokenizer import normalize_text

SENTENCE_COLUMN = "句子"
LABEL_COLUMN = "标签"

_ROW_RE = re.compile(r"<row[^>]*>(.*?)</row>", re.S)
_INLINE_RE = re.compile(r"<is><t[^>]*>(.*?)</t></is>", re.S)


def _cell_values(row_xml: str) -> list[str]:
    return [html.unescape(t) for t in _INLINE_RE.findall(row_xml)]


def read_split(path: str | Path) -> list[str]:
    """Return the stimulus sentences of one ``split_data_N.xlsx`` file."""
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        sheet = archive.read("xl/worksheets/sheet1.xml").decode("utf-8")

    rows = _ROW_RE.findall(sheet)
    if not rows:
        raise ValueError(f"no rows found in {path.name}")

    header = _cell_values(rows[0])
    try:
        col = header.index(SENTENCE_COLUMN)
    except ValueError as exc:
        raise ValueError(
            f"{path.name}: expected a {SENTENCE_COLUMN!r} column, found {header!r}"
        ) from exc

    sentences: list[str] = []
    for row in rows[1:]:
        cells = _cell_values(row)
        if col >= len(cells):
            continue
        text = normalize_text(cells[col])
        if text:
            sentences.append(text)
    return sentences


def split_file_for_run(textdataset_root: str | Path, run: int) -> Path:
    """Map a global run index to its stimulus workbook.

    Run ``N`` corresponds to ``split_data_N.xlsx``; this is the alignment rule
    described in issue #1.
    """
    path = Path(textdataset_root) / f"split_data_{run}.xlsx"
    if not path.exists():
        raise FileNotFoundError(f"no stimulus file for run {run}: expected {path}")
    return path
