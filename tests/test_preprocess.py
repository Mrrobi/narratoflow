"""Tests for the deterministic preprocess layer."""

from __future__ import annotations

from narrato.preprocess import PreprocessConfig, preprocess


def test_whitespace_normalization() -> None:
    res = preprocess("Hei    verden.\n\n\nDet  er  fint.", PreprocessConfig(dedupe_sentences=False))
    assert "    " not in res.text
    assert "\n\n" not in res.text


def test_smart_quotes_normalized() -> None:
    res = preprocess("Han sa “hei” til henne.", PreprocessConfig(dedupe_sentences=False))
    assert "“" not in res.text and "”" not in res.text
    assert '"hei"' in res.text


def test_punct_run_collapse() -> None:
    res = preprocess("Hva!!!! Virkelig????", PreprocessConfig(dedupe_sentences=False))
    assert "!!" not in res.text
    assert "??" not in res.text


def test_dedupe_near_duplicate_sentences() -> None:
    src = (
        "Per gikk til butikken på mandag. "
        "Per gikk til butikken på mandag. "
        "Kari kom hjem etterpå."
    )
    res = preprocess(src, PreprocessConfig(dedupe_sentences=True))
    assert res.removed_sentences >= 1
    assert "Kari" in res.text


def test_stopword_strip_norwegian() -> None:
    src = "Han gikk til butikken og kjøpte et brød."
    res = preprocess(src, PreprocessConfig(strip_stopwords=True, dedupe_sentences=False, lang="no"))
    assert res.stopwords_removed > 0
    assert "butikken" in res.text
    assert "brød" in res.text


def test_unicode_normalization_nfc() -> None:
    # A followed by combining ring above (decomposed Å) -> NFC composes to U+00C5.
    src = "Å"
    res = preprocess(src, PreprocessConfig(dedupe_sentences=False))
    assert "Å" in res.text
    assert "̊" not in res.text
