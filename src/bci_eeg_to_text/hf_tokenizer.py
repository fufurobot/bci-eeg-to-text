"""HuggingFace ``PreTrainedTokenizer`` that wraps the project's CTC vocabulary.

The training code uses :class:`~bci_eeg_to_text.tokenizer.CharTokenizer`, a plain
character vocabulary. This adapter exposes the same mapping through the
HuggingFace tokenizer API so a published folder works with ``AutoTokenizer``
while staying bit-identical to what the model was trained on.

Special tokens follow the CTC convention:

* ``<pad>`` **is** the CTC blank (index 0). Reusing one symbol avoids an
  off-by-one between "padding" and "blank" that silently corrupts training.
* ``<unk>`` maps unseen characters, and is never emitted into a decoded string.
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    from transformers import PreTrainedTokenizer

    _HF = True
except ImportError:  # pragma: no cover
    PreTrainedTokenizer = object  # type: ignore[assignment,misc]
    _HF = False

from bci_eeg_to_text.tokenizer import CharTokenizer

PAD = "<pad>"
UNK_TOKEN = "<unk>"


class EEGCharacterTokenizer(PreTrainedTokenizer):
    """Character-level tokenizer with ``<pad>`` as the CTC blank."""

    vocab_files_names = {"vocab_file": "eeg_vocab.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab: list[str] | None = None, **kwargs) -> None:
        if vocab is None:
            # from_pretrained() supplies the file name from vocab_files_names and
            # passes its path as `vocab_file`; read it back so the loaded
            # tokenizer is identical to the saved one.
            vocab_file = kwargs.pop("vocab_file", None)
            if vocab_file and Path(vocab_file).exists():
                try:
                    vocab = json.loads(Path(vocab_file).read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    vocab = None
        self._vocab_list = list(vocab or [PAD, UNK_TOKEN])
        # from_pretrained re-supplies these from tokenizer_config.json, so only
        # default them here when absent; passing both would be a TypeError.
        kwargs.setdefault("pad_token", PAD)
        kwargs.setdefault("unk_token", UNK_TOKEN)
        super().__init__(**kwargs)

    @property
    def blank_token_id(self) -> int:
        """CTC blank id, which is the padding id by design."""
        return int(self.convert_tokens_to_ids(PAD))

    def get_vocab(self) -> dict[str, int]:
        return {token: i for i, token in enumerate(self._vocab_list)} | dict(
            self.added_tokens_encoder
        )

    def _tokenize(self, text: str) -> list[str]:
        try:
            from bci_eeg_to_text.tokenizer import normalize_text
        except ImportError:  # pragma: no cover - vendored copy

            def normalize_text(value: str) -> str:
                return "".join(value.split())

        return list(normalize_text(text))

    def _convert_token_to_id(self, token: str) -> int:
        return self.get_vocab().get(token, self.get_vocab()[UNK_TOKEN])

    def _convert_id_to_token(self, index: int) -> str:
        if 0 <= index < len(self._vocab_list):
            return self._vocab_list[index]
        return UNK_TOKEN

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens)

    def __len__(self) -> int:
        return len(self._vocab_list)

    @property
    def vocab_size(self) -> int:
        """Size of the base vocabulary.

        ``PreTrainedTokenizer.save_pretrained`` reads this to separate added
        tokens, and the base class raises ``NotImplementedError`` without it.
        """
        return len(self._vocab_list)

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None):
        path = (
            Path(save_directory)
            / (f"{filename_prefix}-" if filename_prefix else "")
            / self.vocab_files_names["vocab_file"]
        )
        path = Path(save_directory) / self.vocab_files_names["vocab_file"]
        path.write_text(json.dumps(self._vocab_list, ensure_ascii=False, indent=2))
        return (str(path),)

    @classmethod
    def build(cls, texts) -> EEGCharacterTokenizer:
        """Build from a corpus of sentences, using the training vocabulary rules.

        Mirrors :meth:`CharTokenizer.build`: index 0 is blank (here ``<pad>``),
        index 1 is unknown, then the sorted characters.
        """
        return cls.from_char_tokenizer(CharTokenizer.build(texts))

    @classmethod
    def from_char_tokenizer(cls, tokenizer: CharTokenizer) -> EEGCharacterTokenizer:
        """Build from the training tokenizer, mapping the reserved symbols."""
        vocab = [PAD, UNK_TOKEN]
        for token in tokenizer.itos[2:]:
            vocab.append(" " if token == " " else token)
        return cls(vocab=vocab)


# Register for the HuggingFace auto classes; the release folder relies on this
# so `AutoTokenizer.from_pretrained(..., trust_remote_code=True)` finds it.
if _HF and hasattr(EEGCharacterTokenizer, "register_for_auto_class"):
    EEGCharacterTokenizer.register_for_auto_class()

#: Name expected by the tests and the release builder.
EEGCharTokenizer = EEGCharacterTokenizer
