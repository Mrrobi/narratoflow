"""Layer 3 — semantic extraction via a small/cheap LLM.

The extractor takes the (already preprocessed + codebook'd) text and asks a
cheap model to fill a Pydantic schema. Output is a compact JSON payload — the
densest representation `narrato` can produce without learned compression.

For sources too long for a single call, use :func:`extract_chunked` to split
the text, extract each chunk independently, then merge the partial payloads
back into a single schema-conformant object.

Async variants (:func:`extract_async`, :func:`extract_chunked_async`) run
chunks concurrently via ``asyncio.gather`` and require the provider to
implement :class:`narrato.providers.AsyncProvider`.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from narrato.providers.base import AsyncProvider, Provider
from narrato.schemas import schema_to_json_schema

_SYSTEM = (
    "You extract facts from source text and emit them as JSON conforming to the "
    "provided schema. Preserve proper nouns and verbatim quotes in the SOURCE language. "
    "Be exhaustive but concise — do not invent details. If a field is unknown, leave it "
    "empty or null. Output only the JSON object, no commentary."
)

_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[A-ZÆØÅa-zæøå0-9])")


# ---------------------------------------------------------------------------
# single-shot extraction
# ---------------------------------------------------------------------------


@dataclass
class ExtractResult:
    payload: dict[str, Any]
    raw_text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    valid: bool = True
    validation_error: str | None = None
    stats: dict = field(default_factory=dict)


def extract(
    text: str,
    schema: type[BaseModel],
    provider: Provider,
    model: str,
    *,
    legend: dict[str, str] | None = None,
    source_lang: str = "no",
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> ExtractResult:
    legend_block = ""
    if legend:
        lines = [f"  {code} = {phrase}" for code, phrase in legend.items()]
        legend_block = "The source text uses these substitution codes:\n" + "\n".join(lines) + "\n\n"

    user = (
        f"{legend_block}"
        f"Source language: {source_lang}\n\n"
        f"SOURCE TEXT:\n```\n{text}\n```\n\n"
        f"Extract the facts and emit them as a JSON object matching the schema."
    )

    json_schema = schema_to_json_schema(schema)
    resp = provider.complete_json(
        system=_SYSTEM,
        user=user,
        model=model,
        schema=json_schema,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    valid = True
    err: str | None = None
    try:
        payload = json.loads(resp.text)
    except Exception as e:
        payload = {"_raw": resp.text}
        valid = False
        err = f"json decode: {e}"

    if valid:
        try:
            schema.model_validate(payload)
        except ValidationError as e:
            valid = False
            err = str(e)

    return ExtractResult(
        payload=payload,
        raw_text=resp.text,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
        model=resp.model,
        valid=valid,
        validation_error=err,
        stats={"chars_out": len(resp.text)},
    )


# ---------------------------------------------------------------------------
# chunked map-reduce extraction
# ---------------------------------------------------------------------------


@dataclass
class ChunkedExtractResult:
    payload: dict[str, Any]
    chunks: int
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    valid: bool = True
    validation_error: str | None = None
    per_chunk: list[ExtractResult] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def extract_chunked(
    text: str,
    schema: type[BaseModel],
    provider: Provider,
    model: str,
    *,
    chunk_chars: int = 8000,
    overlap_chars: int = 200,
    legend: dict[str, str] | None = None,
    source_lang: str = "no",
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> ChunkedExtractResult:
    """Split ``text`` into chunks, extract from each, merge the results.

    Splitting respects sentence boundaries where possible and falls back to a
    hard char cut. ``overlap_chars`` carries the tail of the previous chunk
    into the next, so multi-sentence facts spanning a chunk edge are seen
    twice and de-duplicated by the merge step.
    """

    chunks = _chunk_text(text, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
    per_chunk: list[ExtractResult] = []
    in_tok = 0
    out_tok = 0

    for chunk in chunks:
        res = extract(
            chunk,
            schema=schema,
            provider=provider,
            model=model,
            legend=legend,
            source_lang=source_lang,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        per_chunk.append(res)
        in_tok += res.input_tokens
        out_tok += res.output_tokens

    merged_payload, merge_valid, merge_err = _merge_payloads(
        [r.payload for r in per_chunk], schema=schema
    )

    return ChunkedExtractResult(
        payload=merged_payload,
        chunks=len(chunks),
        input_tokens=in_tok,
        output_tokens=out_tok,
        model=model,
        valid=merge_valid,
        validation_error=merge_err,
        per_chunk=per_chunk,
        stats={
            "chunks": len(chunks),
            "chunk_chars": chunk_chars,
            "overlap_chars": overlap_chars,
            "total_input_tokens": in_tok,
            "total_output_tokens": out_tok,
        },
    )


# ---------------------------------------------------------------------------
# async variants
# ---------------------------------------------------------------------------


async def extract_async(
    text: str,
    schema: type[BaseModel],
    provider: AsyncProvider,
    model: str,
    *,
    legend: dict[str, str] | None = None,
    source_lang: str = "en",
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> ExtractResult:
    legend_block = ""
    if legend:
        lines = [f"  {code} = {phrase}" for code, phrase in legend.items()]
        legend_block = "The source text uses these substitution codes:\n" + "\n".join(lines) + "\n\n"

    user = (
        f"{legend_block}"
        f"Source language: {source_lang}\n\n"
        f"SOURCE TEXT:\n```\n{text}\n```\n\n"
        f"Extract the facts and emit them as a JSON object matching the schema."
    )

    json_schema = schema_to_json_schema(schema)
    resp = await provider.acomplete_json(
        system=_SYSTEM,
        user=user,
        model=model,
        schema=json_schema,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    valid = True
    err: str | None = None
    try:
        payload = json.loads(resp.text)
    except Exception as e:
        payload = {"_raw": resp.text}
        valid = False
        err = f"json decode: {e}"

    if valid:
        try:
            schema.model_validate(payload)
        except ValidationError as e:
            valid = False
            err = str(e)

    return ExtractResult(
        payload=payload,
        raw_text=resp.text,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
        model=resp.model,
        valid=valid,
        validation_error=err,
        stats={"chars_out": len(resp.text)},
    )


async def extract_chunked_async(
    text: str,
    schema: type[BaseModel],
    provider: AsyncProvider,
    model: str,
    *,
    chunk_chars: int = 8000,
    overlap_chars: int = 200,
    legend: dict[str, str] | None = None,
    source_lang: str = "en",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    concurrency: int = 4,
) -> ChunkedExtractResult:
    """Concurrent chunked extraction.

    Runs ``min(concurrency, len(chunks))`` extract calls in parallel via
    ``asyncio.gather``, bounded by a semaphore.
    """

    chunks = _chunk_text(text, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _one(chunk: str) -> ExtractResult:
        async with sem:
            return await extract_async(
                chunk,
                schema=schema,
                provider=provider,
                model=model,
                legend=legend,
                source_lang=source_lang,
                max_tokens=max_tokens,
                temperature=temperature,
            )

    per_chunk: list[ExtractResult] = await asyncio.gather(*[_one(c) for c in chunks])

    merged_payload, merge_valid, merge_err = _merge_payloads(
        [r.payload for r in per_chunk], schema=schema
    )

    in_tok = sum(r.input_tokens for r in per_chunk)
    out_tok = sum(r.output_tokens for r in per_chunk)

    return ChunkedExtractResult(
        payload=merged_payload,
        chunks=len(chunks),
        input_tokens=in_tok,
        output_tokens=out_tok,
        model=model,
        valid=merge_valid,
        validation_error=merge_err,
        per_chunk=per_chunk,
        stats={
            "chunks": len(chunks),
            "chunk_chars": chunk_chars,
            "overlap_chars": overlap_chars,
            "concurrency": concurrency,
            "total_input_tokens": in_tok,
            "total_output_tokens": out_tok,
        },
    )


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------


def _chunk_text(text: str, *, chunk_chars: int, overlap_chars: int) -> list[str]:
    if len(text) <= chunk_chars:
        return [text]

    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    if not sentences:
        return _hard_chunk(text, chunk_chars=chunk_chars, overlap_chars=overlap_chars)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        if current_len + len(sent) + 1 > chunk_chars and current:
            chunks.append(" ".join(current))
            tail = " ".join(current)[-overlap_chars:] if overlap_chars else ""
            current = [tail] if tail else []
            current_len = len(tail)
        current.append(sent)
        current_len += len(sent) + 1

    if current:
        chunks.append(" ".join(current))

    if any(len(c) > chunk_chars * 1.5 for c in chunks):
        return _hard_chunk(text, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
    return chunks


def _hard_chunk(text: str, *, chunk_chars: int, overlap_chars: int) -> list[str]:
    chunks: list[str] = []
    i = 0
    step = max(1, chunk_chars - overlap_chars)
    while i < len(text):
        chunks.append(text[i : i + chunk_chars])
        i += step
    return chunks


def _merge_payloads(
    payloads: list[dict[str, Any]],
    *,
    schema: type[BaseModel],
) -> tuple[dict[str, Any], bool, str | None]:
    """Merge a list of partial payloads into a single schema-conformant object.

    Strategy per field type:
        - list / tuple → concatenate with order-preserving dedupe
        - dict        → recursive merge
        - str / scalar → first non-empty wins
    """
    if not payloads:
        return {}, True, None

    merged: dict[str, Any] = {}
    for p in payloads:
        if not isinstance(p, dict):
            continue
        for key, value in p.items():
            if key not in merged:
                merged[key] = value
            else:
                merged[key] = _merge_values(merged[key], value)

    valid = True
    err: str | None = None
    try:
        schema.model_validate(merged)
    except ValidationError as e:
        valid = False
        err = str(e)
    return merged, valid, err


def _merge_values(a: Any, b: Any) -> Any:
    if a is None or a == "" or a == [] or a == {}:
        return b
    if b is None or b == "" or b == [] or b == {}:
        return a
    if isinstance(a, list) and isinstance(b, list):
        return _dedupe_preserve_order(a + b)
    if isinstance(a, dict) and isinstance(b, dict):
        out: dict[str, Any] = dict(a)
        for k, v in b.items():
            if k in out:
                out[k] = _merge_values(out[k], v)
            else:
                out[k] = v
        return out
    return a


def _dedupe_preserve_order(items: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    out: list[Any] = []
    for item in items:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True) if isinstance(item, dict | list) else item
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out
