"""Provider abstraction. Concrete impls live in sibling modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class ProviderResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    raw: Any = None


class Provider(Protocol):
    name: str

    def complete(
        self,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> ProviderResponse:
        ...

    def complete_json(
        self,
        system: str,
        user: str,
        model: str,
        schema: dict[str, Any] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> ProviderResponse:
        ...


def get_provider(name: str) -> Provider:
    n = name.lower()
    if n == "anthropic":
        from narrato.providers.anthropic import AnthropicProvider

        return AnthropicProvider()
    if n == "openai":
        from narrato.providers.openai import OpenAIProvider

        return OpenAIProvider()
    raise ValueError(f"unknown provider: {name!r}")
