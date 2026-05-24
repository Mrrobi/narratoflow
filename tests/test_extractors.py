"""Offline tests for extract + extract_chunked via MockProvider."""

from __future__ import annotations

import json

from narrato.extractors import extract, extract_chunked
from narrato.providers.mock import MockProvider
from narrato.schemas import NarrativeFacts, NewsFacts


def test_extract_single_valid_payload() -> None:
    canned = {
        "characters": [{"name": "Per", "role": "son", "traits": ["dreamer"]}],
        "setting": {"place": "Hedmark", "time": "1923", "atmosphere": "rural"},
        "events": [{"action": "Per moved to Oslo"}],
        "themes": ["family", "duty"],
        "tone": "wistful",
        "key_quotes": ["Du skulle ha sagt det før"],
    }
    provider = MockProvider(payload=canned)
    res = extract("source text...", schema=NarrativeFacts, provider=provider, model="mock-model")
    assert res.valid is True
    assert res.payload["characters"][0]["name"] == "Per"
    assert provider.calls_complete_json == 1


def test_extract_invalid_payload_flags_validation_error() -> None:
    # Missing required field `action` on Event -> Pydantic validation fails.
    bad = {"events": [{"when": "1923"}]}
    provider = MockProvider(payload=bad)
    res = extract("text", schema=NarrativeFacts, provider=provider, model="mock-model")
    assert res.valid is False
    assert res.validation_error is not None


def test_extract_chunked_splits_and_merges() -> None:
    long_text = ". ".join(f"Sentence number {i}" for i in range(200)) + "."
    payload_template = {
        "characters": [{"name": "Per"}],
        "events": [{"action": "X"}],
        "themes": ["a", "b"],
        "key_quotes": ["quote one"],
    }
    provider = MockProvider(payload=payload_template)
    res = extract_chunked(
        long_text,
        schema=NarrativeFacts,
        provider=provider,
        model="mock-model",
        chunk_chars=500,
        overlap_chars=50,
    )
    assert res.chunks > 1
    assert provider.calls_complete_json == res.chunks
    # Dedupe across chunks: each chunk returns identical payload, merged should still have one of each.
    assert len(res.payload["characters"]) == 1
    assert sorted(res.payload["themes"]) == ["a", "b"]


def test_extract_chunked_dedupes_overlap() -> None:
    text = "Foo. " * 400
    seq = [
        {"events": [{"action": "A"}, {"action": "B"}]},
        {"events": [{"action": "B"}, {"action": "C"}]},
        {"events": [{"action": "C"}, {"action": "D"}]},
    ]
    calls = {"i": 0}

    def per_chunk_payload(_user: str) -> dict:
        out = seq[calls["i"] % len(seq)]
        calls["i"] += 1
        return out

    provider = MockProvider(payload=per_chunk_payload)
    res = extract_chunked(
        text,
        schema=NarrativeFacts,
        provider=provider,
        model="mock-model",
        chunk_chars=200,
        overlap_chars=20,
    )
    actions = [e["action"] for e in res.payload["events"]]
    assert actions == ["A", "B", "C", "D"]


def test_extract_news_schema() -> None:
    payload = {
        "headline": "Big news",
        "what": "Something happened",
        "who": ["Alice", "Bob"],
        "quotes": ["“important quote”"],
    }
    provider = MockProvider(payload=payload)
    res = extract("article text", schema=NewsFacts, provider=provider, model="mock-model")
    assert res.valid is True
    assert res.payload["headline"] == "Big news"


def test_extract_legend_appears_in_prompt() -> None:
    provider = MockProvider(payload={"what": "x"})
    extract(
        "some text",
        schema=NewsFacts,
        provider=provider,
        model="mock-model",
        legend={"§a": "olsen-familien"},
    )
    assert "§a" in provider.last_user
    assert "olsen-familien" in provider.last_user


def test_extract_response_text_is_json() -> None:
    provider = MockProvider(payload={"what": "x"})
    res = extract("t", schema=NewsFacts, provider=provider, model="mock-model")
    parsed = json.loads(res.raw_text)
    assert parsed["what"] == "x"
