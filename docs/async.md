# Async API

`narratoflow` exposes async variants for every step that talks to a provider. They share the same shape as the sync API; just add an `a` prefix.

## Compressor

```python
import asyncio
from narrato import Compressor

c = Compressor.from_profile("rag-en", provider="anthropic")

result = asyncio.run(c.acompress(long_source_text))
```

When the extract layer runs in chunked mode, `acompress()` extracts all chunks **concurrently** via `asyncio.gather` (bounded by a semaphore). Pass `concurrency=N` to tune parallelism:

```python
result = await c.acompress(long_doc, concurrency=8)
```

Free layers (preprocess, codebook) run synchronously inside `acompress()` — they are CPU-bound and fast, so adding a thread pool would slow them down.

## Extractor primitives

```python
from narrato.extractors import extract_async, extract_chunked_async

res = await extract_async(text, schema=MySchema, provider=p, model="...")

res = await extract_chunked_async(
    long_text,
    schema=MySchema,
    provider=p,
    model="...",
    chunk_chars=8000,
    overlap_chars=200,
    concurrency=6,
)
```

## Providers

Every shipped provider implements both sync and async surfaces:

```python
from narrato.providers import get_provider, AsyncProvider

p: AsyncProvider = get_provider("openai")

resp = await p.acomplete_json(system="...", user="...", model="gpt-4o-mini")
```

The protocol is `narrato.providers.AsyncProvider` — type-hint against it when you want async-only callers.

## When to use async

- **Long documents with chunked extraction.** Many small LLM calls run concurrently; you usually see 3–5× speedup on a 10-chunk doc.
- **Multiple documents in a batch.** Wrap `acompress` in an outer `asyncio.gather`.
- **Web servers (FastAPI / Starlette).** Sync `compress` blocks the event loop; `acompress` does not.

Single-shot compression on a small document gains nothing from async — use the sync API for simplicity.

## Provider concurrency limits

Providers have per-key rate limits. Set `concurrency` to a value lower than your tier's parallel requests-per-minute; the default of `4` is safe for most paid tiers but conservative for Tier-4+ usage.
