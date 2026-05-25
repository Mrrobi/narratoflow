"""Anthropic provider adapter (sync + async).

Supports optional prompt caching via Anthropic's ``cache_control`` markers on
the system prompt. Enable by constructing with ``cache=True`` or by setting
``AnthropicProvider.cache`` after init. When enabled, the static system prompt
(which typically contains schema instructions + legend) is marked as an
ephemeral cache breakpoint, so repeated calls within ~5 minutes pay 10 % on
the cached portion.

Cache read tokens are surfaced on
:class:`ProviderResponse.cached_input_tokens`.

See: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
"""

from __future__ import annotations

import json
import os
from typing import Any

from narrato.providers.base import ProviderResponse


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str | None = None, *, cache: bool = False) -> None:
        from anthropic import Anthropic, AsyncAnthropic

        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self._client = Anthropic(api_key=key)
        self._aclient = AsyncAnthropic(api_key=key)
        self.cache = cache

    # ----------------------------------------------------------------- helpers

    def _system_param(self, system: str) -> Any:
        if not self.cache or not system:
            return system or ""
        return [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    @staticmethod
    def _read_usage(resp: Any) -> tuple[int, int, int]:
        usage = resp.usage
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        return input_tokens, output_tokens, cache_read

    @staticmethod
    def _text_from_content(content: Any) -> str:
        return "".join(
            block.text for block in content if getattr(block, "type", None) == "text"
        )

    @staticmethod
    def _emit_payload(content: Any) -> Any:
        for block in content:
            if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == "emit":
                return block.input
        return None

    @staticmethod
    def _json_tools(schema: dict[str, Any] | None) -> list[dict[str, Any]]:
        return [
            {
                "name": "emit",
                "description": "Emit the extracted facts as a JSON object.",
                "input_schema": schema or {"type": "object", "additionalProperties": True},
            }
        ]

    # ----------------------------------------------------------------- sync

    def complete(
        self,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> ProviderResponse:
        resp = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=self._system_param(system),
            messages=[{"role": "user", "content": user}],
        )
        text = self._text_from_content(resp.content)
        i, o, c = self._read_usage(resp)
        return ProviderResponse(
            text=text,
            input_tokens=i,
            output_tokens=o,
            cached_input_tokens=c,
            model=model,
            raw=resp,
        )

    def complete_json(
        self,
        system: str,
        user: str,
        model: str,
        schema: dict[str, Any] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> ProviderResponse:
        resp = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=self._system_param(system),
            tools=self._json_tools(schema),
            tool_choice={"type": "tool", "name": "emit"},
            messages=[{"role": "user", "content": user}],
        )
        payload = self._emit_payload(resp.content)
        if payload is None:
            try:
                payload = json.loads(self._text_from_content(resp.content))
            except Exception:
                payload = {"_raw": self._text_from_content(resp.content)}
        i, o, c = self._read_usage(resp)
        return ProviderResponse(
            text=json.dumps(payload, ensure_ascii=False),
            input_tokens=i,
            output_tokens=o,
            cached_input_tokens=c,
            model=model,
            raw=resp,
        )

    # ----------------------------------------------------------------- async

    async def acomplete(
        self,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> ProviderResponse:
        resp = await self._aclient.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=self._system_param(system),
            messages=[{"role": "user", "content": user}],
        )
        text = self._text_from_content(resp.content)
        i, o, c = self._read_usage(resp)
        return ProviderResponse(
            text=text,
            input_tokens=i,
            output_tokens=o,
            cached_input_tokens=c,
            model=model,
            raw=resp,
        )

    async def acomplete_json(
        self,
        system: str,
        user: str,
        model: str,
        schema: dict[str, Any] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> ProviderResponse:
        resp = await self._aclient.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=self._system_param(system),
            tools=self._json_tools(schema),
            tool_choice={"type": "tool", "name": "emit"},
            messages=[{"role": "user", "content": user}],
        )
        payload = self._emit_payload(resp.content)
        if payload is None:
            try:
                payload = json.loads(self._text_from_content(resp.content))
            except Exception:
                payload = {"_raw": self._text_from_content(resp.content)}
        i, o, c = self._read_usage(resp)
        return ProviderResponse(
            text=json.dumps(payload, ensure_ascii=False),
            input_tokens=i,
            output_tokens=o,
            cached_input_tokens=c,
            model=model,
            raw=resp,
        )
