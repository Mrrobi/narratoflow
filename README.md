# narratoflow

[![PyPI version](https://img.shields.io/pypi/v/narratoflow.svg?logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/narratoflow/)
[![Python versions](https://img.shields.io/pypi/pyversions/narratoflow.svg?logo=python&logoColor=white)](https://pypi.org/project/narratoflow/)
[![CI](https://github.com/Mrrobi/narratoflow/actions/workflows/ci.yml/badge.svg)](https://github.com/Mrrobi/narratoflow/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Tags:** `llm` · `prompt-compression` · `token-optimization` · `cost-reduction` · `anthropic` · `openai` · `ollama` · `claude` · `gpt` · `pydantic` · `python` · `multilingual` · `spacy` · `async` · `rag` · `narrative-generation` · `context-window` · `apache-2.0`

> Compress huge LLM input context into dense intermediate representations. Pay fewer tokens, keep the meaning.

**Docs:** <https://Mrrobi.github.io/narratoflow/> · **PyPI:** <https://pypi.org/project/narratoflow/> · **Source:** <https://github.com/Mrrobi/narratoflow>

`narratoflow` (PyPI name; import as `narrato`) is an open-source Python library (Apache-2.0) for shrinking long source text before sending it to an LLM. It targets any workload where the input dwarfs the output and tokens are the dominant cost — RAG retrieval contexts, narrative generation, transcript summarisation, long-document QA.

The library has a **generic, language- and domain-neutral core**. Common starting points ship as **named profiles** (`rag-en`, `narrative-no`, `news-en`, …) so you do not have to choose every argument up front.

---

## Highlights

- **43% token reduction** on a real Norwegian narrative sample (gpt-4o-mini extractor → gpt-4o target), with **8/10 quality** from an LLM judge
- **3 providers out of the box** — Anthropic, OpenAI, **Ollama** (local models, no API key). Bring your own via a 3-method Protocol.
- **Async API** — `Compressor.acompress(...)` runs chunked extraction concurrently via `asyncio.gather`
- **Layered design** — pick free deterministic layers, an LLM-backed semantic layer, or both
- **Schema-driven** — define a Pydantic model, get a dense JSON payload in return; 5 presets built in (`narrative`, `qa`, `interview`, `dialogue`, `news`)
- **Long-document ready** — automatic chunked map-reduce extraction with overlap-aware merging
- **Anthropic prompt caching** — opt-in via `Compressor(cache=True)`; OpenAI's automatic cache is reported on the response
- **12 bundled languages** — stopwords for en, no, de, fr, es, it, pt, nl, sv, da, fi, pl; auto-detect via `source_lang="auto"`
- **Optional spaCy preprocessing** — POS-aware token stripping that keeps named entities verbatim
- **Typed (PEP 561)** — `py.typed` marker; works directly with mypy / pyright / IDEs
- **Named profiles** — `Compressor.from_profile("rag-en")` for one-line setup; register your own
- **`uv` ready** — pure-PEP-621 hatchling package; `uv add narratoflow`, `uv build`, `uvx` all work

---

## Why

LLM input is priced per token, and a long source document — say a 20-page transcript that feeds a 200-word summary — burns most of the budget *before* the model has written anything.

`narrato` lets you trade a tiny bit of fidelity for a large reduction in input tokens by passing your downstream LLM a dense, machine-friendly representation instead of the raw text. The intermediate representation does not need to be human-readable. It just needs to be:

1. Cheap to produce.
2. Decodable by the downstream LLM into a faithful output.
3. Smaller in tokens than the original.

## Architecture

```
                   ┌──────────────────────────────────────────────┐
   raw text  ──▶   │  L1 preprocess        (deterministic, free)   │
                   │  L2 codebook          (deterministic, free)   │
                   │  L3 semantic extract  (small LLM call)        │
                   │  L4 learned encoder   (planned, v0.6+)        │
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

## Install

```bash
pip install narratoflow
# or
uv add narratoflow
```

Optional extras:

```bash
pip install "narratoflow[lang]"       # langdetect, for source_lang="auto"
pip install "narratoflow[nlp]"        # spaCy, for POS-aware preprocessing
pip install "narratoflow[dev]"        # pytest, ruff, mypy
pip install "narratoflow[docs]"       # mkdocs-material to build the site
pip install "narratoflow[benchmark]"  # rich + tabulate for the bench CLI
```

Set credentials for the provider you use:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
# Ollama: no key — just `ollama pull <model>` and start the daemon.
```

## Quick start

### Using a named profile (recommended)

```python
from narrato import Compressor

c = Compressor.from_profile("rag-en", provider="anthropic")
result = c.compress(long_source_text)

print(result.stats)
# {'input_tokens': 8421, 'output_tokens': 1102, 'ratio': 0.131, ...}
```

Run `narratoflow profiles` to list all built-in profiles, or register your own:

```python
from narrato import Compressor, Profile, register_profile

register_profile(Profile(
    name="legal-en",
    description="English legal documents — chunked + cached",
    source_lang="en",
    schema="qa",
    chunked=True,
    chunk_chars=6000,
    extra={"cache": True},
))

c = Compressor.from_profile("legal-en", provider="anthropic")
```

### Async + concurrent chunked extraction

```python
import asyncio
from narrato import Compressor

c = Compressor.from_profile("long-en", provider="openai")
result = asyncio.run(c.acompress(very_long_document, concurrency=8))
```

### Local models via Ollama

```python
from narrato import Compressor

c = Compressor.from_profile(
    "rag-en",
    provider="ollama",
    extractor_model="llama3",
    target_model="llama3",
)
result = c.compress(text)
```

### Auto-detect language

```python
c = Compressor.from_profile("rag-en", provider="anthropic")
c.source_lang = "auto"           # langdetect if installed, heuristic otherwise

result = c.compress(any_language_document)
print(result.stats["resolved_lang"])     # 'en' / 'no' / 'de' / ...
```

### Explicit construction

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

prompt = Decoder.unpack_prompt(
    result,
    instruction="Skriv en kort fortelling basert på faktene over.",
)
# Send `prompt` to your target LLM.
```

### CLI

```bash
narratoflow profiles
narratoflow schemas
narratoflow compress doc.txt --profile rag-en --out compressed.json
narratoflow eval doc.txt --target-task "Write a 200-word summary." --profile rag-en
```

The `eval` command reports `tokens_in`, `tokens_out`, `ratio`, estimated cost savings, and an LLM-judge quality score (skip with `--skip-quality`).

## Layer reference

| Layer | What it does | Cost | Loss |
|---|---|---|---|
| `preprocess` | Whitespace/punct normalize, near-duplicate sentence dedupe, stopword or spaCy-POS stripping | free | tiny |
| `codebook` | Frequent phrase → short `§x` code, emit legend; tokenizer-aware savings | free | none (with legend) |
| `extract` | Small LLM extracts schema-conformant facts; chunked + concurrent for long docs | cheap LLM call | lossy by design |
| `learned` (v0.6+) | Fine-tuned encoder produces dense codes | one-time train | tunable |

## Schemas

Schemas tell the extractor what to keep. Built-in presets:

| preset | for |
|---|---|
| `narrative` | story / fiction. Characters, setting, ordered events, themes, tone, verbatim quotes. |
| `qa` | fact extraction / RAG. Summary, entities, dates, claims. |
| `interview` | interview / transcript. Interviewer + interviewee, ordered turns, key points, sentiment. |
| `dialogue` | scripted / fictional dialogue. Participants, setting, ordered lines, arc, notable quotes. |
| `news` | news article. Headline, lede, 5W1H (who/what/when/where/why/how), sources, quotes. |

Define your own — any Pydantic v2 `BaseModel`:

```python
from pydantic import BaseModel, Field
from narrato import Compressor

class MyFacts(BaseModel):
    summary: str
    speakers: list[str] = Field(default_factory=list)
    key_dates: list[str] = Field(default_factory=list)

c = Compressor(schema=MyFacts, provider="anthropic")
```

## Providers

| provider | sync | async | JSON mode | prompt cache |
|---|:-:|:-:|---|---|
| Anthropic | ✅ | ✅ | tool-use | opt-in (`cache=True`) |
| OpenAI | ✅ | ✅ | `response_format` JSON schema | automatic (≥1024 tok), reported on `cached_input_tokens` |
| Ollama | ✅ | ✅ | `format=json` + schema reminder | n/a |
| Mock (tests) | ✅ | ✅ | canned payloads | n/a |

Bring your own — implement `complete` + `complete_json` (and optionally `acomplete*`):

```python
from narrato.providers import Provider, ProviderResponse

class MyProvider:
    name = "myco"

    def complete(self, system, user, model, max_tokens=2048, temperature=0.0):
        text = call_my_api(system, user, model)
        return ProviderResponse(text=text, input_tokens=0, output_tokens=0, model=model)

    def complete_json(self, system, user, model, schema=None, max_tokens=2048, temperature=0.0):
        ...

c = Compressor(provider=MyProvider(), ...)
```

## Documentation

Full docs are at <https://Mrrobi.github.io/narratoflow/>. Highlights:

- [Install](https://Mrrobi.github.io/narratoflow/install/) — pip / uv / source / build
- [Quickstart](https://Mrrobi.github.io/narratoflow/quickstart/)
- [Profiles](https://Mrrobi.github.io/narratoflow/profiles/) — bundled presets and custom registration
- [Architecture](https://Mrrobi.github.io/narratoflow/architecture/) — all layers in depth
- [Schemas](https://Mrrobi.github.io/narratoflow/schemas/)
- [Providers](https://Mrrobi.github.io/narratoflow/providers/) — capability matrix
- [Async API](https://Mrrobi.github.io/narratoflow/async/)
- [Language detection](https://Mrrobi.github.io/narratoflow/language/) — `source_lang="auto"`
- [spaCy integration](https://Mrrobi.github.io/narratoflow/spacy/)
- [Type checking](https://Mrrobi.github.io/narratoflow/typing/)
- [Benchmark](https://Mrrobi.github.io/narratoflow/benchmark/)
- [API reference](https://Mrrobi.github.io/narratoflow/api/)
- [Roadmap](https://Mrrobi.github.io/narratoflow/roadmap/)

## Roadmap

- [x] v0.1 — layered preprocess + codebook + schema extract, Anthropic + OpenAI, CLI, eval harness
- [x] v0.2 — chunked map-reduce extraction, Anthropic prompt caching, 3 new schema presets, MockProvider, tokenizer-aware codebook
- [x] v0.3 — generic-core refactor, named profiles, 12-language stopword bundle, CLI `--profile`
- [x] v0.4 — Ollama provider, async API, OpenAI prompt-cache reporting, `py.typed`, `uv build` verified, expanded docs
- [x] v0.5 — optional spaCy preprocessing, language auto-detect, mypy CI job
- [ ] v0.6 — learned encoder R&D, HF Spaces demo, multilingual benchmark corpora, more language stopword sets

See [CHANGELOG.md](CHANGELOG.md) for full release notes.

## Contributing

PRs welcome. Bring your own benchmark.

Local dev:

```bash
git clone https://github.com/Mrrobi/narratoflow.git
cd narratoflow
pip install -e ".[dev]"
pytest -q
mypy narrato
ruff check narrato tests
mkdocs serve     # http://127.0.0.1:8000
```

## License

Apache-2.0. See [LICENSE](LICENSE).
