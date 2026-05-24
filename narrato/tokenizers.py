"""Tokenizer adapters. Unified `count(text)` across providers.

Anthropic does not ship a public offline tokenizer for Claude 4.x; we fall back
to the SDK's `client.messages.count_tokens` when available, otherwise to a
character-heuristic estimate. OpenAI uses `tiktoken` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import tiktoken


class Tokenizer(Protocol):
    name: str

    def count(self, text: str) -> int:
        ...


@dataclass
class OpenAITokenizer:
    """Uses tiktoken. Works fully offline."""

    model: str = "gpt-4o"
    name: str = "openai"

    def __post_init__(self) -> None:
        try:
            self._enc = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self._enc = tiktoken.get_encoding("o200k_base")

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._enc.encode(text))


@dataclass
class AnthropicTokenizer:
    """Uses Anthropic SDK `count_tokens` when available; falls back to estimate.

    The fallback uses a conservative chars-per-token ratio for Latin-script
    languages. For Norwegian text it tends to slightly under-estimate, which
    is acceptable for ratio comparisons.
    """

    model: str = "claude-opus-4-7"
    name: str = "anthropic"
    _client: object | None = None

    def count(self, text: str) -> int:
        if not text:
            return 0
        client = self._get_client()
        if client is not None:
            try:
                resp = client.messages.count_tokens(
                    model=self.model,
                    messages=[{"role": "user", "content": text}],
                )
                return int(resp.input_tokens)
            except Exception:
                pass
        return self._estimate(text)

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import os

            from anthropic import Anthropic

            if not os.getenv("ANTHROPIC_API_KEY"):
                return None
            self._client = Anthropic()
            return self._client
        except Exception:
            return None

    @staticmethod
    def _estimate(text: str) -> int:
        # ~3.8 chars/token for Norwegian/English mix, conservative.
        return max(1, round(len(text) / 3.8))


def get_tokenizer(provider: str, model: str | None = None) -> Tokenizer:
    """Factory by provider name.

    Unknown provider names fall back to the OpenAI tokenizer (BPE via tiktoken)
    so test/mock providers still get a working token counter.
    """
    p = provider.lower()
    if p == "anthropic":
        return AnthropicTokenizer(model=model or "claude-opus-4-7")
    if p == "openai":
        return OpenAITokenizer(model=model or "gpt-4o")
    return OpenAITokenizer(model="gpt-4o")
