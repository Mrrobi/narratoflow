# narrato

> Compress huge LLM input context into dense intermediate representations. Pay fewer tokens, keep the meaning.

`narrato` is an open-source Python library (Apache-2.0) for shrinking long source text before sending it to an LLM. It is designed for tasks like **narrative generation from large source documents**, where the input dwarfs the output and tokens are the dominant cost.

**Provider-agnostic.** Anthropic + OpenAI out of the box.
**Norwegian first-class.** Stopword lists, lemma-friendly preprocessing, Norwegian benchmark samples bundled.
**Pluggable.** Use any layer alone or stack them.

---

## Why

LLM input is priced per token, and a long source document — say a 20-page Norwegian transcript that feeds a 200-word narrative — burns most of the budget *before* the model has written anything.

`narrato` lets you trade a tiny bit of fidelity for a large reduction in input tokens by passing your downstream LLM a dense, machine-friendly representation instead of the raw text.

The intermediate representation does not need to be human-readable. It just needs to be:

1. Cheap to produce.
2. Decodable by the downstream LLM into a faithful narrative.
3. Smaller in tokens than the original.

## Architecture

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

Pick which layers run for each call. Free layers stack with paid layers.

## Quick start

```bash
pip install narrato
```

Set credentials:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
```

### Library

```python
from narrato import Compressor, Decoder

c = Compressor(
    source_lang="no",
    provider="anthropic",
    extractor_model="claude-haiku-4-5-20251001",
    target_model="claude-opus-4-7",
    layers=["preprocess", "codebook", "extract"],
    schema="narrative",
)

result = c.compress(long_norwegian_text)

print(result.stats)
# {'input_tokens': 8421, 'output_tokens': 1102, 'ratio': 0.131, ...}

prompt = Decoder.unpack_prompt(
    result,
    instruction="Skriv en kort fortelling basert på faktene over.",
)
# Send `prompt` to your target LLM.
```

### CLI

```bash
narrato compress input.txt --schema narrative --out compressed.json
narrato eval input.txt --schema narrative --target-task "Skriv en kort fortelling."
```

The eval command reports `tokens_in`, `tokens_out`, `ratio`, estimated cost savings, and an LLM-judge quality score.

## Layer reference

| Layer | What it does | Cost | Loss |
|-------|--------------|------|------|
| `preprocess` | Whitespace/punct normalize, stopword strip, near-duplicate sentence dedupe | free | tiny |
| `codebook` | Frequent phrase → short code, entity → ID rewrite, emit legend | free | none (with legend) |
| `extract` | Small LLM extracts schema-conformant facts | cheap LLM call | lossy by design |
| `learned` (future) | Fine-tuned encoder produces dense codes | one-time train | tunable |

## Schemas

Schemas tell the extractor what to keep. Built-in presets live in `narrato.schemas`:

- `narrative` — characters, setting, ordered events, themes, tone, verbatim quotes
- More to come.

Define your own:

```python
from pydantic import BaseModel
from narrato import Compressor

class MyFacts(BaseModel):
    summary: str
    speakers: list[str]
    key_dates: list[str]

c = Compressor(schema=MyFacts, ...)
```

## Roadmap

- [x] v0.1 — layered preprocess + codebook + schema extract, Anthropic + OpenAI, CLI, eval harness
- [ ] v0.2 — prompt-cache integration on Anthropic path, more schema presets, HF Spaces demo
- [ ] v0.3 — local model support (Ollama), Norwegian spaCy pipeline integration
- [ ] v0.4 — learned encoder (fine-tuned small model, distributed via HuggingFace)

## Contributing

PRs welcome. Bring your own benchmark.

## License

Apache-2.0. See [LICENSE](LICENSE).
