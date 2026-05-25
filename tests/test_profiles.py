"""Profile registry and Compressor.from_profile tests."""

from __future__ import annotations

import pytest

from narrato import Compressor
from narrato.profiles import (
    Profile,
    get_profile,
    list_profiles,
    register_profile,
    unregister_profile,
)
from narrato.providers.mock import MockProvider


def test_bundled_profiles_present() -> None:
    names = {p.name for p in list_profiles()}
    for required in {"rag-en", "qa-en", "narrative-en", "narrative-no", "rag-no", "news-en"}:
        assert required in names, f"missing bundled profile: {required}"


def test_get_profile_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_profile("does-not-exist")


def test_from_profile_applies_defaults() -> None:
    c = Compressor.from_profile("narrative-no", provider=MockProvider())
    assert c.source_lang == "no"
    assert c.schema == "narrative"


def test_from_profile_overrides_win() -> None:
    c = Compressor.from_profile(
        "rag-en",
        provider=MockProvider(),
        source_lang="de",
        target_model="custom-model",
    )
    assert c.source_lang == "de"
    assert c.target_model == "custom-model"
    assert c.schema == "qa"


def test_rag_profile_enables_stopword_strip() -> None:
    c = Compressor.from_profile("rag-en", provider=MockProvider())
    assert c.preprocess_config is not None
    assert c.preprocess_config.strip_stopwords is True


def test_qa_profile_does_not_strip_stopwords() -> None:
    c = Compressor.from_profile("qa-en", provider=MockProvider())
    # qa-en explicitly leaves stopwords on (better for QA accuracy)
    assert c.preprocess_config is None


def test_long_profile_enables_chunking() -> None:
    c = Compressor.from_profile("long-en", provider=MockProvider())
    assert c.chunked is True
    assert c.chunk_chars >= 1000


def test_register_and_use_custom_profile() -> None:
    profile = Profile(
        name="legal-en-test",
        description="Test legal profile",
        source_lang="en",
        schema="qa",
        extra={"cache": True},
    )
    register_profile(profile)
    try:
        c = Compressor.from_profile("legal-en-test", provider=MockProvider())
        assert c.source_lang == "en"
        assert c.cache is True
    finally:
        unregister_profile("legal-en-test")


def test_register_duplicate_requires_overwrite() -> None:
    profile = Profile(name="dup-test", description="dup")
    register_profile(profile)
    try:
        with pytest.raises(ValueError):
            register_profile(profile)
        register_profile(profile, overwrite=True)
    finally:
        unregister_profile("dup-test")


def test_default_compressor_is_generic() -> None:
    """No profile, no provider call — should not assume Norwegian or narrative."""
    c = Compressor(provider="anthropic", layers=["preprocess", "codebook"])
    assert c.source_lang == "en"
    assert c.schema == "qa"
