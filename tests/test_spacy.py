"""spaCy integration tests — skipped when spaCy is not installed."""

from __future__ import annotations

import pytest

spacy = pytest.importorskip("spacy")  # noqa: F841

from narrato.preprocess import PreprocessConfig, preprocess  # noqa: E402
from narrato.spacy_pipeline import is_available, model_for_lang  # noqa: E402


def test_is_available_true() -> None:
    assert is_available() is True


def test_model_for_lang_known() -> None:
    assert model_for_lang("en") == "en_core_web_sm"
    assert model_for_lang("de") == "de_core_news_sm"
    assert model_for_lang("no") == "nb_core_news_sm"


def test_model_for_lang_unknown_defaults_to_english() -> None:
    assert model_for_lang("zz") == "en_core_web_sm"


@pytest.mark.skipif(
    not pytest.importorskip("spacy").util.is_package("en_core_web_sm"),
    reason="en_core_web_sm model not installed",
)
def test_preprocess_uses_spacy_when_configured() -> None:
    text = "Hello world. This is a test. Another sentence here."
    cfg = PreprocessConfig(spacy_model="en_core_web_sm", dedupe_sentences=True)
    res = preprocess(text, cfg)
    assert res.stats["spacy_used"] is True
