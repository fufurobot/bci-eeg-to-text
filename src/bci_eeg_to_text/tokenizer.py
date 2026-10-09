"""Character-level tokenizer for Chinese EEG-to-text decoding.

The tokenizer defines the label space used by the CTC head. Two invariants
matter for correctness and are enforced here:

1. Index ``blank`` (always 0) is reserved for the CTC blank symbol and is
   never assigned to a real character.
2. The vocabulary is deterministic: it is built from a fixed, sorted set of
   characters, so a checkpoint's output layer always matches the vocabulary
   that was used to train it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

BLANK = "\x00"
UNK = "\ufffd"

#: Punctuation observed in the Chisco stimulus sentences (split_data_*.xlsx).
DEFAULT_PUNCTUATION = "，。？！、；："


def normalize_text(text: str) -> str:
    """Normalize a stimulus sentence into the tokenizer's canonical form.

    Chinese stimulus text mixes full-width and ASCII punctuation and may carry
    stray whitespace. Normalizing first keeps the vocabulary small, which
    directly reduces the CTC output dimension.
    """
    table = str.maketrans(
        {
            ",": "，",
            ".": "。",
            "?": "？",
            "!": "！",
            ";": "；",
            ":": "：",
            "、": "、",
            "\u3000": "",
        }
    )
    text = text.translate(table)
    # Drop all whitespace: the corpus contains no meaningful spaces.
    text = "".join(text.split())
    return text


@dataclass(frozen=True)
class CharTokenizer:
    """Maps characters to contiguous integer ids for CTC training.

    ``itos[blank]`` is the CTC blank. Real characters occupy indices
    ``1..len(vocab)``.
    """

    itos: tuple[str, ...]
    stoi: dict[str, int] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not self.itos:
            raise ValueError("tokenizer vocabulary must not be empty")
        if self.itos[0] != BLANK:
            raise ValueError("index 0 must be the CTC blank symbol")
        if len(self.itos) < 2 or self.itos[1] != UNK:
            raise ValueError("index 1 must be the UNK symbol")
        # Check the reserved-symbol rules before the generic duplicate check so
        # that a reused blank/UNK reports the specific cause.
        if BLANK in self.itos[1:]:
            raise ValueError("blank must never be a real vocabulary entry")
        if UNK in self.itos[2:]:
            raise ValueError("UNK must never be a real vocabulary entry")
        if len(set(self.itos)) != len(self.itos):
            raise ValueError("tokenizer vocabulary contains duplicates")
        stoi = {c: i for i, c in enumerate(self.itos)}
        object.__setattr__(self, "stoi", stoi)

    @property
    def blank(self) -> int:
        """Index of the CTC blank symbol."""
        return 0

    @property
    def unk(self) -> int:
        """Index returned for out-of-vocabulary characters."""
        return self.stoi[UNK]

    def __len__(self) -> int:
        """Number of classes, including blank and UNK."""
        return len(self.itos)

    @classmethod
    def build(cls, texts: Iterable[str]) -> CharTokenizer:
        """Build a vocabulary from a corpus of sentences.

        Indices: ``0`` blank, ``1`` UNK, then real characters sorted for
        determinism.
        """
        chars: set[str] = set()
        for text in texts:
            chars.update(normalize_text(text))
        chars.discard(BLANK)
        chars.discard(UNK)
        if not chars:
            raise ValueError(
                "tokenizer vocabulary must not be empty: corpus contains no "
                "characters other than the reserved blank/UNK symbols"
            )
        # sorted() gives a stable ordering independent of corpus iteration order.
        itos = (BLANK, UNK, *sorted(chars))
        return cls(itos=itos)

    def encode(self, text: str) -> list[int]:
        """Encode text to ids, mapping unknown characters to UNK."""
        return [self.stoi.get(c, self.unk) for c in normalize_text(text)]

    def decode(self, ids: Sequence[int]) -> str:
        """Decode ids back to text, skipping blank and UNK."""
        out: list[str] = []
        for i in ids:
            if i == self.blank or i == self.unk:
                continue
            out.append(self.itos[i])
        return "".join(out)

    def save(self, path: str | Path) -> None:
        """Persist the vocabulary as JSON."""
        Path(path).write_text(
            json.dumps({"itos": list(self.itos)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> CharTokenizer:
        """Load a vocabulary previously written by :meth:`save`."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(itos=tuple(data["itos"]))
