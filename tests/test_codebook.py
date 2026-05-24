"""Tests for codebook layer."""

from __future__ import annotations

from narrato.codebook import CodebookConfig, build, decode


def test_frequent_multiword_phrase_replaced_with_code() -> None:
    text = (
        "Den røde låven sto i Hedmark. "
        "Den røde låven var gammel. "
        "Den røde låven trengte maling."
    )
    res = build(text, CodebookConfig(min_phrase_words=2, min_occurrences=3, min_phrase_chars=6))
    assert res.legend, "expected at least one legend entry"
    assert any(("røde låven" in v.lower() or "den røde" in v.lower()) for v in res.legend.values())
    assert "§" in res.text


def test_rare_phrase_not_replaced() -> None:
    text = "Per gikk til butikken. Kari kom hjem etterpå. Solen skinte."
    res = build(text, CodebookConfig(min_occurrences=3))
    assert res.legend == {}
    assert res.text == text


def test_round_trip_decode() -> None:
    text = (
        "Per og Kari bodde på gården. "
        "Per og Kari hadde mange dyr. "
        "Per og Kari var lykkelige."
    )
    res = build(text, CodebookConfig(min_occurrences=3, min_phrase_words=3, min_phrase_chars=6))
    assert res.legend, "expected legend entry for repeated 3-word phrase"
    restored = decode(res.text, res.legend)
    for phrase in res.legend.values():
        assert phrase in restored.lower()


def test_legend_string_format() -> None:
    text = "alpha beta gamma. alpha beta gamma. alpha beta gamma."
    res = build(text, CodebookConfig(min_occurrences=3, min_phrase_words=2, min_phrase_chars=5))
    assert res.legend
    ls = res.legend_string()
    assert ls.startswith("LEGEND:")
    for code in res.legend:
        assert code in ls


def test_safe_strip_existing_code_prefix() -> None:
    text = "§a should be stripped from input."
    res = build(text, CodebookConfig(min_occurrences=99))  # no legend will be built
    assert "§" not in res.text
