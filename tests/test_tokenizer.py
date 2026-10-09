"""Tests for the character-level CTC tokenizer."""

from __future__ import annotations

import json

import pytest

from bci_eeg_to_text.tokenizer import BLANK, UNK, CharTokenizer, normalize_text


class TestNormalizeText:
    def test_converts_ascii_punctuation_to_fullwidth(self) -> None:
        assert normalize_text("你好,世界?") == "你好，世界？"

    def test_strips_all_whitespace(self) -> None:
        assert normalize_text(" 你好 世界\n") == "你好世界"

    def test_is_idempotent(self) -> None:
        once = normalize_text("你好,世界!")
        assert normalize_text(once) == once


class TestBuild:
    def test_blank_is_index_zero(self) -> None:
        tok = CharTokenizer.build(["你好"])
        assert tok.blank == 0
        assert tok.itos[0] == BLANK

    def test_unk_is_index_one_but_never_decoded(self) -> None:
        tok = CharTokenizer.build(["你好"])
        assert tok.unk == 1
        assert tok.decode(tok.encode("你好")) == "你好"

    def test_vocabulary_is_deterministic_regardless_of_order(self) -> None:
        a = CharTokenizer.build(["你好世界", "abc"])
        b = CharTokenizer.build(["abc", "世界你好"])
        assert a.itos == b.itos

    def test_normalizes_before_building(self) -> None:
        tok = CharTokenizer.build(["你好,世界"])
        # The full-width comma is in the vocabulary; the ASCII one is not.
        assert "，" in tok.stoi
        assert "," not in tok.stoi

    def test_rejects_empty_corpus(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            CharTokenizer.build([""])

    def test_build_rejects_corpus_yielding_only_reserved_symbols(self) -> None:
        """A corpus of only blank/UNK leaves no real classes to predict."""
        with pytest.raises(ValueError, match="must not be empty"):
            CharTokenizer.build([BLANK, UNK])


class TestEncodeDecode:
    def test_roundtrip_preserves_sentence(self) -> None:
        sentence = "我想预订去广州的机票"
        tok = CharTokenizer.build([sentence, "附近还有其他的酒店吗"])
        assert tok.decode(tok.encode(sentence)) == sentence

    def test_unknown_character_maps_to_unk_and_is_dropped_on_decode(self) -> None:
        tok = CharTokenizer.build(["你好"])
        ids = tok.encode("你好Z")
        assert ids[-1] == tok.unk
        assert tok.decode(ids) == "你好"

    def test_blank_ids_are_skipped_when_decoding(self) -> None:
        tok = CharTokenizer.build(["你好"])
        ids = [tok.blank, tok.stoi["你"], tok.blank, tok.stoi["好"], tok.blank]
        assert tok.decode(ids) == "你好"

    def test_encode_applies_normalization(self) -> None:
        tok = CharTokenizer.build(["你好，世界"])
        assert tok.encode("你好,世界") == tok.encode("你好，世界")

    def test_len_includes_blank_and_unk(self) -> None:
        tok = CharTokenizer.build(["你好"])
        # blank + unk + 2 characters
        assert len(tok) == 4

    def test_every_real_class_is_reachable(self) -> None:
        """Each vocabulary character must have a unique, non-blank index."""
        tok = CharTokenizer.build(["你好世界"])
        indices = [tok.stoi[c] for c in "你好世界"]
        assert len(set(indices)) == len(indices)
        assert min(indices) >= 2


class TestValidation:
    def test_rejects_vocabulary_not_starting_with_blank(self) -> None:
        with pytest.raises(ValueError, match="index 0 must be the CTC blank"):
            CharTokenizer(itos=("a", "b"))

    def test_rejects_duplicate_characters(self) -> None:
        with pytest.raises(ValueError, match="duplicates"):
            CharTokenizer(itos=(BLANK, UNK, "a", "a"))

    def test_rejects_unk_reused_as_a_real_entry(self) -> None:
        with pytest.raises(ValueError, match="UNK must never be"):
            CharTokenizer(itos=(BLANK, UNK, "a", UNK))

    def test_rejects_missing_unk(self) -> None:
        with pytest.raises(ValueError, match="index 1 must be the UNK"):
            CharTokenizer(itos=(BLANK, "a", "b"))

    def test_rejects_blank_reused_as_a_real_entry(self) -> None:
        with pytest.raises(ValueError, match="blank must never be"):
            CharTokenizer(itos=(BLANK, UNK, "a", BLANK))


class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path) -> None:
        tok = CharTokenizer.build(["你好世界", "abc"])
        path = tmp_path / "vocab.json"
        tok.save(path)
        assert CharTokenizer.load(path).itos == tok.itos

    def test_saved_file_is_readable_utf8_json(self, tmp_path) -> None:
        """Vocabulary must be stored unescaped so it is human-inspectable."""
        tok = CharTokenizer.build(["你好"])
        path = tmp_path / "vocab.json"
        tok.save(path)
        raw = path.read_text(encoding="utf-8")
        assert "你" in raw
        assert json.loads(raw)["itos"][0] == BLANK
