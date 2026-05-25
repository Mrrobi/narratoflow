# Example notebooks

Practical notebooks. No marketing fluff. Each one prints real numbers — tokens, latency, cost — so you can decide whether `narratoflow` is worth wiring into your own pipeline.

| notebook | needs API key | what it shows |
|---|---|---|
| [`01_quickstart.ipynb`](01_quickstart.ipynb) | no | minimum viable use; free layers + `MockProvider`. Runs offline. |
| [`02_token_savings.ipynb`](02_token_savings.ipynb) | **yes** (OpenAI) | real OpenAI call on a 4 KB document; measured before/after tokens + USD cost |
| [`03_long_doc_async.ipynb`](03_long_doc_async.ipynb) | no | chunked extraction; sync vs async wall-clock timing with a delay-injected mock provider |
| [`04_custom_schema.ipynb`](04_custom_schema.ipynb) | no | define a domain schema (meeting notes), watch the extractor fill it; mock provider with realistic payload |

Honest expectations:

- `narratoflow` is a *token reducer*, not a magic eraser. On a 500-token toy doc you will save very little money. The library shines on inputs ≥ 5 000 tokens where input cost dominates.
- Free layers (preprocess + codebook) alone usually save 2–10 %. The extract layer is where the big wins live, but it costs a small LLM call.
- Quality is **lossy by design**. If you need lossless context, do not use the extract layer — use preprocess + codebook only.

## Run them

```bash
pip install -e ".[dev]"      # editable install from repo root
pip install jupyterlab       # one-time
jupyter lab examples/notebooks/
```

For `02_token_savings.ipynb`:

```bash
export OPENAI_API_KEY=sk-...
```

All other notebooks run fully offline.
