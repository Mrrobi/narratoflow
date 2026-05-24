# Architecture

`narratoflow` is a **layered compressor**. Each layer is independently usable and the orchestrator (`Compressor`) chains them in order.

```
                   ┌──────────────────────────────────────────────┐
   raw text  ──▶   │  L1 preprocess        (deterministic, free)   │
                   │  L2 codebook          (deterministic, free)   │
                   │  L3 semantic extract  (small LLM call)        │
                   │  L4 learned encoder   (optional, future)      │
                   └──────────────────────────────────────────────┘
                                      │
                                      ▼
                          CompressionResult
                          (payload + legend + stats)
                                      │
                                      ▼
                   ┌──────────────────────────────────────────────┐
                   │  Decoder.unpack_prompt()                      │
                   │  → ready-to-send prompt for downstream LLM    │
                   └──────────────────────────────────────────────┘
```

## L1 — preprocess (free)

Deterministic text cleanup. No LLM calls. See `narrato.preprocess`.

| operation | default |
|---|---|
| Unicode NFC normalization | on |
| Whitespace collapse | on |
| Smart-quote canonicalisation | on |
| Punctuation-run collapse (`!!!` → `!`) | on |
| Near-duplicate sentence dedupe (Jaccard ≥ 0.85) | on |
| Stopword stripping | off by default for narrative use; turn on for fact extraction |

```python
from narrato.preprocess import preprocess, PreprocessConfig

res = preprocess(text, PreprocessConfig(lang="no", strip_stopwords=False))
```

## L2 — codebook (free)

Builds a per-document codebook of frequent multi-word phrases, replaces each occurrence with a short `§code`, and emits a legend.

Codes only win when occurrence × (phrase length − code length) exceeds legend cost. The library uses character length as a fast proxy and the pipeline measures real tokens afterwards.

```python
from narrato.codebook import build, CodebookConfig

res = build(text, CodebookConfig(min_phrase_words=2, min_occurrences=3))
print(res.legend)             # {"§a": "olsen-familien"}
print(res.legend_string())    # ready to prepend to a prompt
```

## L3 — semantic extract (cheap LLM call)

A small, cheap LLM (e.g. `claude-haiku-4-5-20251001` or `gpt-4o-mini`) is given the (already preprocessed + codebook'd) text and a Pydantic schema, and asked to fill the schema. The filled object is the dense payload sent downstream.

Output is JSON. The downstream model receives the JSON instead of the raw text, plus its task instruction.

```python
from narrato.extractors import extract
from narrato.providers import get_provider
from narrato.schemas import NarrativeFacts

res = extract(
    text,
    schema=NarrativeFacts,
    provider=get_provider("openai"),
    model="gpt-4o-mini",
)
```

This is the layer where the bulk of compression happens — typically 3–10x reduction depending on schema density and source redundancy.

### Chunked map-reduce (v0.2+)

When the source exceeds a single-call budget, the pipeline switches to map-reduce automatically (or set `Compressor(chunked=True)` to force it).

```python
c = Compressor(chunked=True, chunk_chars=8000, overlap_chars=200, ...)
```

Splitting respects sentence boundaries where possible; `overlap_chars` carries the tail of one chunk into the next so multi-sentence facts spanning a chunk edge are seen twice and deduplicated by the merger.

Merge strategy per field type:

- list / tuple → concatenate with order-preserving dedupe
- dict → recursive merge
- str / scalar → first non-empty wins

### Prompt caching (v0.2+, Anthropic only)

```python
c = Compressor(provider="anthropic", cache=True, ...)
```

Marks the system prompt (which carries schema instructions + legend) as an ephemeral cache breakpoint. Repeated calls within the cache TTL (~5 min) pay 10 % on the cached portion. No-op on OpenAI for now.

## L4 — learned encoder (planned)

Future versions will ship an optional fine-tuned small model (Phi-3, Qwen-1.5B class) that produces dense codes directly, without requiring a per-document LLM call. Distributed via Hugging Face.

## CompressionResult

The orchestrator returns:

```python
@dataclass
class CompressionResult:
    payload: str                 # the compressed payload (text or JSON-as-string)
    legend: dict[str, str]       # codebook legend; empty if L2 not used
    format: str                  # "text" or "json"
    layers_run: list[str]
    stats: dict[str, Any]        # tokens before/after per layer, ratio, etc.
```

## Decoder

`Decoder` assembles the final prompt to send downstream:

```python
prompt = Decoder.unpack_prompt(result, instruction="…")
# or split for caching:
msgs = Decoder.unpack_messages(result, instruction="…")
# msgs = {"system": "<legend>", "user": "<payload + instruction>"}
```

[Schemas →](schemas.md){ .md-button .md-button--primary }
