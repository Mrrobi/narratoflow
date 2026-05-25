# Quickstart

## Install

```bash
pip install narratoflow
```

## Credentials

`narratoflow` reads provider keys from the standard environment variables:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
```

Or use a `.env` file with `python-dotenv` in your own code:

```python
from dotenv import load_dotenv
load_dotenv()
```

## Library — one-liner via a profile

```python
from narrato import Compressor, Decoder

c = Compressor.from_profile("rag-en", provider="anthropic")
result = c.compress(long_source_text)
```

See [Profiles](profiles.md) for the full list of bundled profiles.

## Library — explicit construction

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
    target_lang="no",
)
# Send `prompt` to your target LLM.
```

## OpenAI provider

Same API, different `provider` argument:

```python
c = Compressor(
    source_lang="no",
    provider="openai",
    extractor_model="gpt-4o-mini",
    target_model="gpt-4o",
    schema="narrative",
)
```

## CLI

```bash
narratoflow compress input.txt --schema narrative --out compressed.json

narratoflow eval input.txt \
  --target-task "Skriv en kort fortelling." \
  --provider openai \
  --extractor-model gpt-4o-mini \
  --target-model gpt-4o
```

The eval command reports tokens in/out, estimated cost savings, and an LLM-judge quality score (skipped with `--skip-quality`).

## Free-layer-only mode

To shave tokens without any LLM cost, run just the deterministic layers:

```python
c = Compressor(layers=["preprocess", "codebook"], provider="anthropic")
result = c.compress(text)
# Modest savings — but zero extra API calls.
```

[Architecture →](architecture.md){ .md-button .md-button--primary }
