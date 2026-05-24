"""Pydantic schemas that drive the semantic extractor.

A schema describes *what to keep* from the source text. The extractor LLM is
asked to fill it; the filled object is the dense payload sent downstream.

Custom schemas: pass any ``BaseModel`` subclass to ``Compressor(schema=...)``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Character(BaseModel):
    name: str
    role: str | None = None
    traits: list[str] = Field(default_factory=list)


class Setting(BaseModel):
    place: str | None = None
    time: str | None = None
    atmosphere: str | None = None


class Event(BaseModel):
    when: str | None = None
    who: list[str] = Field(default_factory=list)
    action: str
    outcome: str | None = None


class NarrativeFacts(BaseModel):
    """Dense fact bundle suitable for narrative generation."""

    characters: list[Character] = Field(default_factory=list)
    setting: Setting = Field(default_factory=Setting)
    events: list[Event] = Field(default_factory=list)
    themes: list[str] = Field(default_factory=list)
    tone: str | None = None
    key_quotes: list[str] = Field(
        default_factory=list,
        description="Verbatim quotes worth preserving in the source language.",
    )


class QAFacts(BaseModel):
    """Schema for QA / fact-extraction flows."""

    summary: str
    entities: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    claims: list[str] = Field(default_factory=list)


PRESETS: dict[str, type[BaseModel]] = {
    "narrative": NarrativeFacts,
    "qa": QAFacts,
}


def get_schema(name_or_class: str | type[BaseModel]) -> type[BaseModel]:
    if isinstance(name_or_class, str):
        if name_or_class not in PRESETS:
            raise ValueError(
                f"unknown schema preset {name_or_class!r}; available: {list(PRESETS)}"
            )
        return PRESETS[name_or_class]
    return name_or_class


def schema_to_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic model to a JSON Schema dict for provider JSON modes."""
    return model.model_json_schema()
