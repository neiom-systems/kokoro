"""
Luxembourgish phonemization utilities backed by espeak-ng.

The G2P used here mirrors the pathway KPipeline employs for other
espeak-backed languages, keeping the output compatible with the
Kokoro phoneme vocabulary and its 510-symbol constraint per chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional

from misaki import espeak

MAX_PHONEME_LEN = 510
DEFAULT_CHUNK_SIZE = 400
SENTENCE_SPLIT_REGEX = re.compile(r"([.!?]+)")
SPACE_REGEX = re.compile(r"\s+")
TOKEN_REGEX = re.compile(r"\s+|[A-Za-zÀ-ÖØ-öø-ÿ]+(?:['-][A-Za-zÀ-ÖØ-öø-ÿ]+)*|[^\w\s]", re.UNICODE)
WORD_REGEX = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:['-][A-Za-zÀ-ÖØ-öø-ÿ]+)*", re.UNICODE)

CUSTOM_LEXICON = {
    "moien": "ˈmɔɪ̯ən",
    "alleguer": "aləˈɡuːɐ̯",
    "wéi": "vɛɪ̯",
    "geet": "ˈɡeːt",
    "et": "ət",
    "merci": "ˈmɛʁsi",
    "villmools": "fɪlˈmoːls",
}


@dataclass
class PhonemizedSegment:
    """Container for a normalized text segment and its phoneme sequence."""

    text: str
    phonemes: str


class LuxembourgishPhonemizer:
    """
    Lightweight wrapper around espeak's Luxembourgish rules.

    Usage:
        phonemizer = LuxembourgishPhonemizer()
        segments = phonemizer("Moien alleguer, wéi geet et?")
    """

    def __init__(self, language: str = "lb", chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
        try:
            self._g2p = espeak.EspeakG2P(language=language)
        except Exception as exc:  # pragma: no cover - only hit when espeak is misconfigured
            raise RuntimeError(
                f"Failed to initialize espeak G2P for language '{language}'. "
                "Ensure espeak-ng is installed with Luxembourgish support."
            ) from exc
        self._chunk_size = chunk_size

    @staticmethod
    def _normalize(text: str) -> str:
        """Trim whitespace and collapse runs of spaces."""
        return SPACE_REGEX.sub(" ", text.strip())

    def _chunk_text(self, text: str) -> List[str]:
        """
        Break the text into manageable pieces before G2P.

        Preference is given to sentence boundaries; if a chunk still
        exceeds the configured size we fall back to raw character spans.
        """
        if len(text) <= self._chunk_size:
            return [text]

        sentences = SENTENCE_SPLIT_REGEX.split(text)
        chunks: List[str] = []
        current = ""

        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]
            sentence = sentence.strip()
            if not sentence:
                continue

            candidate = f"{current} {sentence}".strip() if current else sentence
            if current and len(candidate) > self._chunk_size:
                chunks.append(current)
                current = sentence
            else:
                current = candidate

        if current:
            chunks.append(current)

        if not chunks:
            return [text]

        refined: List[str] = []
        for chunk in chunks:
            if len(chunk) <= self._chunk_size:
                refined.append(chunk)
                continue
            for start in range(0, len(chunk), self._chunk_size):
                piece = chunk[start : start + self._chunk_size].strip()
                if piece:
                    refined.append(piece)
        return refined or [text]

    @staticmethod
    def _lookup_custom(word: str) -> Optional[str]:
        return CUSTOM_LEXICON.get(word.casefold())

    def _g2p_word(self, word: str) -> str:
        phonemes, _ = self._g2p(word)
        phonemes = phonemes.strip()
        return self._postprocess_phonemes(phonemes)

    @staticmethod
    def _postprocess_phonemes(phonemes: str) -> str:
        replacements = {
            "oj": "ɔɪ̯",
            "ɜɪ": "ɛɪ̯",
            "ʒ": "ɡ",
            "ʁ": "ɐ̯",
            "ɑ": "a",
        }
        for src, dest in replacements.items():
            phonemes = phonemes.replace(src, dest)
        phonemes = re.sub(r"([^\sˈ])ˈ", r"ˈ\1", phonemes)
        return phonemes

    def _phonemize_chunk(self, chunk: str) -> str:
        tokens = TOKEN_REGEX.findall(chunk)
        parts: List[str] = []
        previous_was_word = False

        for token in tokens:
            if not token:
                continue
            if token.isspace():
                previous_was_word = previous_was_word
                continue
            if WORD_REGEX.fullmatch(token):
                phoneme = self._lookup_custom(token)
                if phoneme is None:
                    phoneme = self._g2p_word(token)
                if parts and previous_was_word:
                    parts.append(" ")
                parts.append(phoneme)
                previous_was_word = True
            else:
                parts.append(token)
                previous_was_word = False

        phoneme_str = "".join(parts).strip()
        if len(phoneme_str) > MAX_PHONEME_LEN:
            phoneme_str = phoneme_str[:MAX_PHONEME_LEN]
        return phoneme_str

    def phonemize_text(self, text: str) -> List[PhonemizedSegment]:
        """Return phonemized segments for a single utterance."""
        normalized = self._normalize(text)
        if not normalized:
            return []

        segments: List[PhonemizedSegment] = []
        for chunk in self._chunk_text(normalized):
            phonemes = self._phonemize_chunk(chunk)
            if phonemes:
                segments.append(PhonemizedSegment(text=chunk, phonemes=phonemes))
        return segments

    def __call__(self, text: str) -> List[PhonemizedSegment]:
        return self.phonemize_text(text)

    def phonemize_lines(self, lines: Iterable[str]) -> Iterator[PhonemizedSegment]:
        """Stream phonemized segments for an iterable of utterances."""
        for line in lines:
            yield from self.phonemize_text(line)
