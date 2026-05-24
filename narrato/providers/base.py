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


def get_provider(name: str, *, cache: bool = False) -> Provider:
    """Factory by provider name.

    Parameters
    ----------
    name:
        ``"anthropic"`` or ``"openai"``.
    cache:
        Enable provider-side prompt caching when supported. Currently honored
        by the Anthropic provider (marks the system prompt as an ephemeral
        cache breakpoint). Ignored by providers without native caching.
    """
    n = name.lower()
    if n == "anthropic":
        from narrato.providers.anthropic import AnthropicProvider

        return AnthropicProvider(cache=cache)
    if n == "openai":
        from narrato.providers.openai import OpenAIProvider

        return OpenAIProvider()
    raise ValueError(f"unknown provider: {name!r}")
