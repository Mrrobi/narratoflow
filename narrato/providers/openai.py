"""OpenAI provider adapter."""

from __future__ import annotations

import json
import os
from typing import Any

from narrato.providers.base import ProviderResponse


class OpenAIProvider:
    name = "openai"

    def __init__(self, api_key: str | None = None) -> None:
        from openai import OpenAI

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self._client = OpenAI(api_key=key)

    def complete(
        self,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> ProviderResponse:
        resp = self._client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system or ""},
                {"role": "user", "content": user},
            ],
        )
        choice = resp.choices[0].message.content or ""
        usage = resp.usage
        return ProviderResponse(
            text=choice,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
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
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system or ""},
                {"role": "user", "content": user},
            ],
        }
        if schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "extraction",
                    "schema": schema,
                    "strict": False,
                },
            }
        else:
            kwargs["response_format"] = {"type": "json_object"}

        resp = self._client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content or "{}"
        try:
            payload = json.loads(content)
        except Exception:
            payload = {"_raw": content}
        usage = resp.usage
        return ProviderResponse(
            text=json.dumps(payload, ensure_ascii=False),
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=model,
            raw=resp,
        )
