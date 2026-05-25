# Providers

`narratoflow` ships adapters for three providers and a Mock for tests. All adapters implement the same surface so the pipeline does not care which one you pick.

| provider | sync | async | streaming | JSON mode | prompt cache |
|---|:-:|:-:|:-:|:-:|:-:|
| Anthropic | ✅ | ✅ | not in narrato | tool-use | opt-in (`cache=True`) |
| OpenAI | ✅ | ✅ | not in narrato | `response_format` JSON schema | automatic (≥1024 tok) |
| Ollama | ✅ | ✅ | not in narrato | `format=json` | n/a |
| Mock | ✅ | ✅ | n/a | canned payloads | n/a |

## Anthropic

```python
from narrato.providers import get_provider
p = get_provider("anthropic", cache=True)
```

- Reads `ANTHROPIC_API_KEY` from env.
- JSON mode uses a forced tool call named ``emit`` so the model must emit a structured payload.
- When `cache=True`, the system prompt (schema instructions + legend) is marked as an ephemeral cache breakpoint (5-min TTL). Cache reads are surfaced on `ProviderResponse.cached_input_tokens`.

## OpenAI

```python
from narrato.providers import get_provider
p = get_provider("openai")
```

- Reads `OPENAI_API_KEY` from env.
- JSON mode uses `response_format = {"type": "json_schema", ...}` (loose strictness) so any Pydantic schema works.
- OpenAI's prompt cache is automatic for prompts ≥ 1024 tokens. The adapter reads `usage.prompt_tokens_details.cached_tokens` and exposes it on `ProviderResponse.cached_input_tokens`.

## Ollama (local models)

```python
from narrato.providers import get_provider
p = get_provider("ollama")  # uses OLLAMA_HOST or http://localhost:11434
```

- Requires a running Ollama daemon. `ollama pull <model>` first.
- JSON mode uses `format=json` plus a system-prompt schema reminder; small open models may produce loose JSON — the pipeline reports a `validation_error` instead of crashing.
- No API key required.

Example with the Compressor:

```python
from narrato import Compressor
c = Compressor(
    provider="ollama",
    extractor_model="llama3",
    target_model="llama3",
    source_lang="en",
    schema="qa",
)
```

## Mock (testing)

```python
from narrato.providers import MockProvider
from narrato import Compressor

mock = MockProvider(payload={"summary": "...", "claims": ["..."]})
c = Compressor(provider=mock, schema="qa", layers=["extract"])
```

`MockProvider` accepts a static dict, a callable that takes the user prompt and returns a payload, or a sequence of canned responses. It tracks `calls_complete` and `calls_complete_json` so tests can assert on number of LLM calls.

## Bring your own provider

Subclass-by-protocol — implement `complete`, `complete_json` (and optionally the `acomplete*` async variants for `Compressor.acompress` support):

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
