"""Schema preset and conversion tests."""

from __future__ import annotations

from pydantic import BaseModel

from narrato.schemas import NarrativeFacts, get_schema, schema_to_json_schema


def test_preset_resolution() -> None:
    assert get_schema("narrative") is NarrativeFacts


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
