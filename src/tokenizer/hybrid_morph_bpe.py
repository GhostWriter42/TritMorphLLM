"""Hybrid morphology-aware BPE tokenizer.

This module trains a lightweight BPE tokenizer while biasing segmentation toward
simple morpheme boundaries. The implementation is intentionally transparent and
easy to extend with richer analyzers later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import BpeTrainer


DEFAULT_PREFIXES: tuple[str, ...] = (
    "anti",
    "auto",
    "counter",
    "de",
    "dis",
    "fore",
    "hyper",
    "il",
    "im",
    "in",
    "inter",
    "micro",
    "mis",
    "non",
    "over",
    "post",
    "pre",
    "re",
    "semi",
    "sub",
    "super",
    "trans",
    "un",
    "under",
)

DEFAULT_SUFFIXES: tuple[str, ...] = (
    "ability",
    "able",
    "ation",
    "ed",
    "en",
    "er",
    "est",
    "ful",
    "hood",
    "ible",
    "ing",
    "ion",
    "ish",
    "ism",
    "ist",
    "ity",
    "ive",
    "less",
    "ly",
    "ment",
    "ness",
    "ous",
    "s",
    "ship",
    "tion",
    "ward",
    "wards",
    "y",
)

SPECIAL_TOKENS: list[str] = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]


@dataclass(slots=True)
class WordSpan:
    """Span mapping from original words to flattened subword indices."""

    word: str
    start: int
    end: int


@dataclass(slots=True)
class TokenizedExample:
    """Tokenized example with span mapping for later composition."""

    input_ids: List[int]
    attention_mask: List[int]
    word_spans: List[WordSpan]
    words: List[str]
    tokens: List[str]
    word_ids: List[int] | None = None


class HybridTokenizer:
    """Morphology-aware BPE tokenizer wrapper.

    The tokenizer trains a base BPE model on morphologically segmented text and
    preserves a mapping between words and the produced subword pieces.
    """

    def __init__(
        self,
        vocab_size: int = 8192,
        lowercase: bool = True,
        min_frequency: int = 2,
        max_word_length: int = 64,
        prefixes: Sequence[str] | None = None,
        suffixes: Sequence[str] | None = None,
    ) -> None:
        self.vocab_size = vocab_size
        self.lowercase = lowercase
        self.min_frequency = min_frequency
        self.max_word_length = max_word_length
        self.prefixes = tuple(sorted(prefixes or DEFAULT_PREFIXES, key=len, reverse=True))
        self.suffixes = tuple(sorted(suffixes or DEFAULT_SUFFIXES, key=len, reverse=True))

        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = WhitespaceSplit()
        self._is_trained = False
        self._word_pattern = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]")

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.token_to_id("[PAD]") or 0

    @property
    def bos_token_id(self) -> int:
        return self.tokenizer.token_to_id("[BOS]") or 2

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.token_to_id("[EOS]") or 3

    @property
    def unk_token_id(self) -> int:
        return self.tokenizer.token_to_id("[UNK]") or 1

    @property
    def vocab_size_actual(self) -> int:
        return self.tokenizer.get_vocab_size()

    def word_to_id(self, word: str) -> int | None:
        return self.tokenizer.token_to_id(self._normalize_word(word))

    def train_from_iterator(self, texts: Iterable[str]) -> None:
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=SPECIAL_TOKENS,
            continuing_subword_prefix="##",
        )
        self.tokenizer.train_from_iterator(self._iter_morph_corpus(texts), trainer=trainer)
        self._is_trained = True

    def save(self, path: str | Path) -> None:
        self.tokenizer.save(str(path))

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        vocab_size: int = 8192,
        lowercase: bool = True,
        min_frequency: int = 2,
        max_word_length: int = 64,
    ) -> "HybridTokenizer":
        instance = cls(
            vocab_size=vocab_size,
            lowercase=lowercase,
            min_frequency=min_frequency,
            max_word_length=max_word_length,
        )
        instance.tokenizer = Tokenizer.from_file(str(path))
        instance._is_trained = True
        return instance

    def tokenize_word(self, word: str) -> List[str]:
        self._require_trained()
        normalized = self._normalize_word(word)
        if not normalized:
            return []

        pieces = self._segment_word(normalized)
        tokens: List[str] = []
        for piece in pieces:
            encoding = self.tokenizer.encode(piece, add_special_tokens=False)
            current_tokens = encoding.tokens if encoding.tokens else ["[UNK]"]
            tokens.extend(current_tokens)
        return tokens

    def encode(self, text: str, add_special_tokens: bool = False) -> TokenizedExample:
        self._require_trained()
        words = self._split_text(text)
        input_ids: List[int] = []
        attention_mask: List[int] = []
        word_spans: List[WordSpan] = []
        flat_tokens: List[str] = []

        if add_special_tokens:
            input_ids.append(self.bos_token_id)
            attention_mask.append(1)
            flat_tokens.append("[BOS]")

        for raw_word in words:
            subwords = self.tokenize_word(raw_word)
            start = len(input_ids)
            if not subwords:
                continue
            for token in subwords:
                token_id = self.tokenizer.token_to_id(token)
                input_ids.append(self.unk_token_id if token_id is None else token_id)
                attention_mask.append(1)
                flat_tokens.append(token)
            end = len(input_ids)
            word_spans.append(WordSpan(word=raw_word, start=start, end=end))

        if add_special_tokens:
            input_ids.append(self.eos_token_id)
            attention_mask.append(1)
            flat_tokens.append("[EOS]")

        return TokenizedExample(
            input_ids=input_ids,
            attention_mask=attention_mask,
            word_spans=word_spans,
            words=words,
            tokens=flat_tokens,
            word_ids=[self.word_to_id(word) or self.unk_token_id for word in words],
        )

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=True)

    def _iter_morph_corpus(self, texts: Iterable[str]) -> Iterator[str]:
        for text in texts:
            words = self._split_text(text)
            if not words:
                continue
            segmented_words = [" ".join(self._segment_word(self._normalize_word(word))) for word in words]
            yield " ".join(piece for piece in segmented_words if piece)

    def _split_text(self, text: str) -> List[str]:
        return [match.group(0) for match in self._word_pattern.finditer(text)]

    def _normalize_word(self, word: str) -> str:
        normalized = word.lower() if self.lowercase else word
        return normalized[: self.max_word_length]

    def _segment_word(self, word: str) -> List[str]:
        if len(word) <= 3 or not word.isalpha():
            return [word]

        segments = [word]
        prefix = self._match_prefix(word)
        if prefix:
            stem = word[len(prefix) :]
            if len(stem) >= 3:
                segments = [prefix, stem]

        expanded: List[str] = []
        for segment in segments:
            suffixes = self._extract_suffix_chain(segment)
            if suffixes:
                stem, suffix_chain = suffixes
                if stem:
                    expanded.append(stem)
                expanded.extend(suffix_chain)
            else:
                expanded.append(segment)
        return [segment for segment in expanded if segment]

    def _match_prefix(self, word: str) -> str | None:
        for prefix in self.prefixes:
            if word.startswith(prefix) and len(word) - len(prefix) >= 3:
                return prefix
        return None

    def _extract_suffix_chain(self, word: str) -> tuple[str, List[str]] | None:
        stem = word
        chain: List[str] = []
        while len(stem) >= 4:
            matched = None
            for suffix in self.suffixes:
                if stem.endswith(suffix) and len(stem) - len(suffix) >= 3:
                    matched = suffix
                    break
            if matched is None:
                break
            stem = stem[: -len(matched)]
            chain.insert(0, matched)
        if chain:
            return stem, chain
        return None

    def _require_trained(self) -> None:
        if not self._is_trained:
            raise RuntimeError("HybridTokenizer must be trained before use.")
