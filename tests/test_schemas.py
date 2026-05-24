"""Schema preset and conversion tests."""

from __future__ import annotations

from pydantic import BaseModel

from narrato.schemas import (
    DialogueFacts,
    InterviewFacts,
    NarrativeFacts,
    NewsFacts,
    QAFacts,
    get_schema,
    list_presets,
    schema_to_json_schema,
)


def test_preset_resolution() -> None:
    assert get_schema("narrative") is NarrativeFacts
    assert get_schema("qa") is QAFacts
    assert get_schema("interview") is InterviewFacts
    assert get_schema("dialogue") is DialogueFacts
    assert get_schema("news") is NewsFacts


def test_list_presets_returns_all() -> None:
    presets = list_presets()
    assert "narrative" in presets
    assert "interview" in presets
    assert "news" in presets
    assert len(presets) >= 5


def test_passthrough_class() -> None:
    class Custom(BaseModel):
        summary: str

    assert get_schema(Custom) is Custom


def test_unknown_preset_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        get_schema("does-not-exist")


def test_json_schema_roundtrip() -> None:
    js = schema_to_json_schema(NarrativeFacts)
    assert js["type"] == "object"
    assert "properties" in js
    assert "characters" in js["properties"]


def test_news_schema_5w1h_fields() -> None:
    js = schema_to_json_schema(NewsFacts)
    for field in ("who", "what", "when", "where", "why", "how"):
        assert field in js["properties"]


def test_interview_schema_turns_structure() -> None:
    js = schema_to_json_schema(InterviewFacts)
    assert "turns" in js["properties"]
