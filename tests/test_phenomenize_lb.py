import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover - initialization guard
    sys.path.insert(0, str(ROOT))

from kokoro.train.phenomenize_lb import (  # noqa: E402
    LuxembourgishPhonemizer,
    MAX_PHONEME_LEN,
    PhonemizedSegment,
)


def _skip_if_espeak_missing() -> LuxembourgishPhonemizer:
    try:
        return LuxembourgishPhonemizer()
    except RuntimeError as exc:  # pragma: no cover - environment without espeak
        pytest.skip(str(exc))


@pytest.fixture(scope="session")
def sample_store(pytestconfig):
    return pytestconfig._lux_samples


def test_phonemize_text_returns_segment():
    phonemizer = _skip_if_espeak_missing()

    segments = phonemizer("Moien alleguer")

    assert segments and len(segments) == 1
    segment = segments[0]
    assert isinstance(segment, PhonemizedSegment)
    assert segment.text == "Moien alleguer"
    assert segment.phonemes
    assert len(segment.phonemes) <= MAX_PHONEME_LEN


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("Moien alleguer", "ˈmɔɪ̯ən aləˈɡuːɐ̯"),
        ("Wéi geet et?", "vɛɪ̯ ˈɡeːt ət?"),
        ("Merci villmools", "ˈmɛʁsi fɪlˈmoːls"),
    ],
)
def test_known_phrases_phonemes(phrase: str, expected: str, sample_store):
    phonemizer = _skip_if_espeak_missing()

    segments = phonemizer(phrase)

    assert segments and len(segments) == 1
    assert segments[0].phonemes == expected
    sample_store.append((phrase, segments[0].phonemes))


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("Villmools Merci fir deng Hëllef", "fɪlˈmoːls ˈmɛʁsi ˈfiːɐ̯ ˈdeŋ ˈhələf"),
        ("Ech ginn an d'Stad fir akafen", "ˈeç ˈgin ˈan dzˈtaːd ˈfiːɐ̯ aˈkaːfən"),
        ("Haut ass et kal a sonneg", "ˈhaʊt ˈas ət ˈkaːl a ˈzonəç"),
        ("Kanns de muer owes kommen?", "ˈkanz də ˈmuːɐ̯ ˈoːvəs ˈkomən?"),
        ("Mir ginn op d'Kirmes mam Frënd", "ˈmiːɐ̯ ˈgin ˈop dˈkiʁməs mam fʁənt"),
    ],
)
def test_additional_sentences(phrase: str, expected: str, sample_store):
    phonemizer = _skip_if_espeak_missing()

    segments = phonemizer(phrase)

    assert segments and len(segments) == 1
    assert segments[0].phonemes == expected
    sample_store.append((phrase, segments[0].phonemes))


def test_chunking_respects_configured_size():
    try:
        tiny_chunker = LuxembourgishPhonemizer(chunk_size=20)
    except RuntimeError as exc:  # pragma: no cover - environment without espeak
        pytest.skip(str(exc))

    text = " ".join(["Dëst ass eng zimlech laang Luxemburgesch Saz."] * 5)
    segments = tiny_chunker(text)

    assert segments, "expected chunking to return segments"
    for segment in segments:
        assert len(segment.text) <= 20
        assert segment.phonemes
        assert len(segment.phonemes) <= MAX_PHONEME_LEN
