"""OpenAI provider adapter (sync + async).

OpenAI's prompt caching is automatic for prompts >= 1024 tokens. When a cache
hit occurs, the API returns ``usage.prompt_tokens_details.cached_tokens``.
This adapter surfaces that value on :class:`ProviderResponse.cached_input_tokens`.
"""

from __future__ import annotations

import json
import os
from typing import Any

from narrato.providers.base import ProviderResponse


def _cached_tokens(usage: Any) -> int:
    if not usage:
        return 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is None:
        return 0
    return int(getattr(details, "cached_tokens", 0) or 0)


class OpenAIProvider:
    name = "openai"

    def __init__(self, api_key: str | None = None) -> None:
        from openai import AsyncOpenAI, OpenAI

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self._client = OpenAI(api_key=key)
        self._aclient = AsyncOpenAI(api_key=key)

    # ----------------------------------------------------------------- sync

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
        return _to_response(resp, model, json_mode=False)

    def complete_json(
        self,
        system: str,
        user: str,
        model: str,
        schema: dict[str, Any] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> ProviderResponse:
        kwargs = _json_kwargs(system, user, model, schema, max_tokens, temperature)
        resp = self._client.chat.completions.create(**kwargs)
        return _to_response(resp, model, json_mode=True)

    # ----------------------------------------------------------------- async

    async def acomplete(
        self,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> ProviderResponse:
        resp = await self._aclient.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system or ""},
                {"role": "user", "content": user},
            ],
        )
        return _to_response(resp, model, json_mode=False)

    async def acomplete_json(
        self,
        system: str,
        user: str,
        model: str,
        schema: dict[str, Any] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> ProviderResponse:
        kwargs = _json_kwargs(system, user, model, schema, max_tokens, temperature)
        resp = await self._aclient.chat.completions.create(**kwargs)
        return _to_response(resp, model, json_mode=True)


def _json_kwargs(
    system: str,
    user: str,
    model: str,
    schema: dict[str, Any] | None,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
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
            "json_schema": {"name": "extraction", "schema": schema, "strict": False},
        }
    else:
        kwargs["response_format"] = {"type": "json_object"}
    return kwargs


def _to_response(resp: Any, model: str, *, json_mode: bool) -> ProviderResponse:
    content = (resp.choices[0].message.content or "") if not json_mode else (
        resp.choices[0].message.content or "{}"
    )
    if json_mode:
        try:
            payload = json.loads(content)
        except Exception:
            payload = {"_raw": content}
        text = json.dumps(payload, ensure_ascii=False)
    else:
        text = content

    usage = resp.usage
    return ProviderResponse(
        text=text,
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
        cached_input_tokens=_cached_tokens(usage),
        model=model,
        raw=resp,
    )
