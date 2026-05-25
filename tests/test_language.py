"""Language auto-detection tests (heuristic + langdetect when available)."""

from __future__ import annotations

from narrato.language import _heuristic, detect, supported


def test_supported_includes_bundled() -> None:
    s = supported()
    for code in ("en", "no", "de", "fr", "es", "fi"):
        assert code in s


def test_detect_short_text_returns_default() -> None:
    assert detect("hi", default="xx") == "xx"


def test_heuristic_english_text() -> None:
    text = "The quick brown fox jumps over the lazy dog. This is an English sentence with several common words."
    assert _heuristic(text, default="xx") == "en"


def test_heuristic_norwegian_text() -> None:
    text = (
        "Olsen-familien bodde på en gård i Hedmark. "
        "De hadde drevet gården i tre generasjoner. "
        "Per var den eldste sønnen og skulle overta gården."
    )
    assert _heuristic(text, default="xx") == "no"


def test_heuristic_german_text() -> None:
    text = (
        "Die Familie wohnte auf einem Bauernhof in Bayern. "
        "Sie hatten den Hof drei Generationen lang bewirtschaftet. "
        "Der älteste Sohn sollte den Hof übernehmen."
    )
    assert _heuristic(text, default="xx") == "de"


def test_heuristic_french_text() -> None:
    text = (
        "La famille habitait dans une ferme en Bretagne. "
        "Ils avaient exploité la ferme pendant trois générations. "
        "Le fils aîné devait reprendre la ferme."
    )
    assert _heuristic(text, default="xx") == "fr"


def test_detect_falls_back_to_default_for_garbage() -> None:
    assert detect("xxx yyy zzz qqq", default="en") == "en"


def test_compressor_auto_detects() -> None:
    """Compressor with source_lang='auto' resolves to detected lang in stats."""
    from narrato import Compressor

    c = Compressor(provider="anthropic", source_lang="auto", layers=["preprocess"])
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "Another sentence here for good measure."
    )
    result = c.compress(text)
    assert result.stats["resolved_lang"] == "en"
