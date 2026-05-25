"""Ollama provider — local LLMs via the Ollama HTTP API.

Tested against Ollama 0.x. The provider talks to ``/api/chat`` and uses
Ollama's ``format`` option for structured JSON output. Models that lack tool
or JSON-mode training (e.g. small base models) may produce loose JSON; the
extractor wraps the response and the pipeline reports validation errors
without crashing.

Install Ollama: https://ollama.com — then ``ollama pull llama3``.

This provider has no separate Python SDK requirement: it uses ``httpx`` which
is already pulled in as a transitive dependency of ``anthropic`` and
``openai``. Set ``OLLAMA_HOST`` (default: ``http://localhost:11434``).
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from narrato.providers.base import ProviderResponse


class OllamaProvider:
    name = "ollama"

    def __init__(
        self,
        host: str | None = None,
        *,
        timeout: float = 120.0,
    ) -> None:
        self.host = (host or os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(base_url=self.host, timeout=timeout)
        self._aclient = httpx.AsyncClient(base_url=self.host, timeout=timeout)

    def __del__(self) -> None:  # best-effort
        client = getattr(self, "_client", None)
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _approx_tokens(text: str) -> int:
        # Ollama doesn't always return prompt/completion token counts; estimate.
        return max(1, round(len(text) / 4))

    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        resp = self._client.post("/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def _apost_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        resp = await self._aclient.post("/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _build_chat_payload(
        system: str,
        user: str,
        model: str,
        max_tokens: int,
        temperature: float,
        *,
        json_format: bool = False,
        schema: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], str]:
        messages: list[dict[str, str]] = []
        sys_prompt = system or ""
        if json_format and schema is not None:
            sys_prompt = (
                (sys_prompt + "\n\n" if sys_prompt else "")
                + "Respond ONLY with a JSON object matching this JSON Schema:\n"
                + json.dumps(schema, ensure_ascii=False)
            )
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": user})
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if json_format:
            payload["format"] = "json"
        return payload, sys_prompt

    # ---------------------------------------------------------------- complete

    # ----------------------------------------------------------------- sync

    def complete(
        self,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> ProviderResponse:
        payload, sys_prompt = self._build_chat_payload(
            system, user, model, max_tokens, temperature, json_format=False
        )
        body = self._post_chat(payload)
        return self._to_response(body, model, sys_prompt, user, json_mode=False)

    def complete_json(
        self,
        system: str,
        user: str,
        model: str,
        schema: dict[str, Any] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> ProviderResponse:
        payload, sys_prompt = self._build_chat_payload(
            system, user, model, max_tokens, temperature, json_format=True, schema=schema
        )
        body = self._post_chat(payload)
        return self._to_response(body, model, sys_prompt, user, json_mode=True)

    # ----------------------------------------------------------------- async

    async def acomplete(
        self,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> ProviderResponse:
        payload, sys_prompt = self._build_chat_payload(
            system, user, model, max_tokens, temperature, json_format=False
        )
        body = await self._apost_chat(payload)
        return self._to_response(body, model, sys_prompt, user, json_mode=False)

    async def acomplete_json(
        self,
        system: str,
        user: str,
        model: str,
        schema: dict[str, Any] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> ProviderResponse:
        payload, sys_prompt = self._build_chat_payload(
            system, user, model, max_tokens, temperature, json_format=True, schema=schema
        )
        body = await self._apost_chat(payload)
        return self._to_response(body, model, sys_prompt, user, json_mode=True)

    def _to_response(
        self,
        body: dict[str, Any],
        model: str,
        sys_prompt: str,
        user: str,
        *,
        json_mode: bool,
    ) -> ProviderResponse:
        text = body.get("message", {}).get("content", "") or ("{}" if json_mode else "")
        if json_mode:
            try:
                payload_obj = json.loads(text)
            except Exception:
                payload_obj = {"_raw": text}
            text = json.dumps(payload_obj, ensure_ascii=False)
        prompt_tokens = body.get("prompt_eval_count") or self._approx_tokens(sys_prompt + user)
        completion_tokens = body.get("eval_count") or self._approx_tokens(text)
        return ProviderResponse(
            text=text,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            model=model,
            raw=body,
        )
