"""Async API tests using the MockProvider (no network)."""

from __future__ import annotations

import asyncio

import pytest

from narrato import Compressor
from narrato.extractors import extract_async, extract_chunked_async
from narrato.providers.mock import MockProvider
from narrato.schemas import NewsFacts


def _run(coro):
    return asyncio.run(coro)


def test_extract_async_single() -> None:
    provider = MockProvider(payload={"what": "happened"})

    async def go():
        return await extract_async(
            "text", schema=NewsFacts, provider=provider, model="m"
        )

    res = _run(go())
    assert res.valid is True
    assert res.payload["what"] == "happened"
    assert provider.calls_complete_json == 1


def test_extract_chunked_async_runs_concurrently() -> None:
    text = ". ".join(f"S {i}" for i in range(400)) + "."
    provider = MockProvider(payload={"what": "x"})

    async def go():
        return await extract_chunked_async(
            text,
            schema=NewsFacts,
            provider=provider,
            model="m",
            chunk_chars=200,
            overlap_chars=20,
            concurrency=4,
        )

    res = _run(go())
    assert res.chunks > 1
    assert provider.calls_complete_json == res.chunks


def test_compressor_acompress() -> None:
    provider = MockProvider(payload={"what": "hello"})
    c = Compressor(
        provider=provider,
        layers=["preprocess", "extract"],
        schema="news",
    )

    async def go():
        return await c.acompress("Source text. Another sentence.")

    res = _run(go())
    assert res.format == "json"
    assert res.stats["extract"]["mode"] in {"single-async", "chunked-async"}


def test_compressor_acompress_chunked() -> None:
    long_text = ". ".join(f"Sent {i}" for i in range(500)) + "."
    provider = MockProvider(payload={"what": "fact"})
    c = Compressor(
        provider=provider,
        layers=["extract"],
        schema="news",
        chunk_chars=400,
        overlap_chars=20,
    )

    async def go():
        return await c.acompress(long_text, concurrency=3)

    res = _run(go())
    assert res.stats["extract"]["mode"] == "chunked-async"
    assert res.stats["extract"]["chunks"] > 1


def test_acompress_skips_async_when_no_extract_layer() -> None:
    """Free-layer-only acompress is still legal and returns a result without LLM calls."""
    provider = MockProvider()
    c = Compressor(provider=provider, layers=["preprocess", "codebook"])

    async def go():
        return await c.acompress("Hei verden. Dette er en test setning.")

    res = _run(go())
    assert "extract" not in res.layers_run
    assert provider.calls_complete_json == 0


@pytest.mark.parametrize("concurrency", [1, 2, 8])
def test_concurrency_does_not_break(concurrency: int) -> None:
    text = ". ".join(f"X{i}" for i in range(120))
    provider = MockProvider(payload={"what": "a"})
    c = Compressor(provider=provider, layers=["extract"], schema="news", chunk_chars=200)

    async def go():
        return await c.acompress(text, concurrency=concurrency)

    res = _run(go())
    assert res.stats["extract"]["valid"] is True
