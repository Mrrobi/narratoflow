"""Anthropic provider adapter."""

from __future__ import annotations

import json
import os
from typing import Any

from narrato.providers.base import ProviderResponse


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str | None = None) -> None:
        from anthropic import Anthropic

        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self._client = Anthropic(api_key=key)

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
            system=system or "",
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        return ProviderResponse(
            text=text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
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
        """Anthropic has no native JSON mode; we use a tool-call to force a structured object."""
        tool_schema = schema or {"type": "object", "additionalProperties": True}
        tools = [
            {
                "name": "emit",
                "description": "Emit the extracted facts as a JSON object.",
                "input_schema": tool_schema,
            }
        ]
        resp = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or "",
            tools=tools,
            tool_choice={"type": "tool", "name": "emit"},
            messages=[{"role": "user", "content": user}],
        )
        payload: Any = None
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == "emit":
                payload = block.input
                break
        if payload is None:
            text = "".join(
                b.text for b in resp.content if getattr(b, "type", None) == "text"
            )
            try:
                payload = json.loads(text)
            except Exception:
                payload = {"_raw": text}
        return ProviderResponse(
            text=json.dumps(payload, ensure_ascii=False),
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            model=model,
            raw=resp,
        )
