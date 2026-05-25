"""Ollama provider tests using httpx mock transports (no network)."""

from __future__ import annotations

import asyncio
import json

import httpx

from narrato.providers.ollama import OllamaProvider


def _mock_response(content: str, *, prompt_eval_count: int = 10, eval_count: int = 5):
    body = {
        "model": "llama3",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
    }
    return httpx.Response(200, json=body)


def _attach_mock(provider: OllamaProvider, handler):
    transport = httpx.MockTransport(handler)
    provider._client = httpx.Client(base_url=provider.host, transport=transport)
    provider._aclient = httpx.AsyncClient(base_url=provider.host, transport=transport)


def test_ollama_complete_text() -> None:
    provider = OllamaProvider()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        body = json.loads(request.content.decode())
        assert body["model"] == "llama3"
        assert "messages" in body
        return _mock_response("hello from llama")

    _attach_mock(provider, handler)
    resp = provider.complete(system="sys", user="hi", model="llama3")
    assert resp.text == "hello from llama"
    assert resp.input_tokens == 10
    assert resp.output_tokens == 5


def test_ollama_complete_json_forces_format_json() -> None:
    provider = OllamaProvider()
    sent_payload: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sent_payload
        sent_payload = json.loads(request.content.decode())
        return _mock_response('{"what": "thing"}')

    _attach_mock(provider, handler)
    resp = provider.complete_json(
        system="sys",
        user="hi",
        model="llama3",
        schema={"type": "object", "properties": {"what": {"type": "string"}}},
    )
    assert sent_payload["format"] == "json"
    assert json.loads(resp.text) == {"what": "thing"}


def test_ollama_acomplete_async() -> None:
    provider = OllamaProvider()

    def handler(_request: httpx.Request) -> httpx.Response:
        return _mock_response("async hello")

    _attach_mock(provider, handler)

    async def go():
        return await provider.acomplete(system="", user="hi", model="llama3")

    resp = asyncio.run(go())
    assert resp.text == "async hello"


def test_ollama_handles_missing_token_counts() -> None:
    provider = OllamaProvider()

    def handler(_request: httpx.Request) -> httpx.Response:
        body = {
            "model": "llama3",
            "message": {"role": "assistant", "content": "ok"},
            "done": True,
        }
        return httpx.Response(200, json=body)

    _attach_mock(provider, handler)
    resp = provider.complete(system="", user="hi", model="llama3")
    assert resp.input_tokens > 0
    assert resp.output_tokens > 0
