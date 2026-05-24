# Benchmark

`narratoflow` ships a reproducible benchmark harness so you can measure compression ratio and downstream quality on your own data.

## Bundled sample

`benchmarks/samples/norwegian_sample.txt` — a ~500-word Norwegian short story used as the canonical Norwegian-narrative benchmark.

## Run

```bash
narratoflow eval benchmarks/samples/norwegian_sample.txt \
  --target-task "Skriv en kort fortelling (200-300 ord) på norsk basert på fakta." \
  --provider openai \
  --extractor-model gpt-4o-mini \
  --target-model gpt-4o
```

Or via the Python API:

```python
from narrato.benchmark import run_benchmark
from narrato.pipeline import Compressor

c = Compressor(provider="openai", extractor_model="gpt-4o-mini",
               target_model="gpt-4o", schema="narrative")
report = run_benchmark(text, instruction="Skriv en kort fortelling.",
                      compressor=c, target_model="gpt-4o")
print(report.to_json())
```

## Output fields

| field | meaning |
|---|---|
| `tokens_source` | tokens in the raw source, measured for the target model |
| `tokens_compressed` | tokens in the final downstream prompt (legend + payload + instruction) |
| `ratio` | `tokens_compressed / tokens_source` — lower is better |
| `cost_baseline` | estimated USD cost for the baseline narrative generation (full source) |
| `cost_compressed` | estimated USD cost for the compressed run (incl. extractor call) |
| `cost_savings_pct` | percent saved versus baseline |
| `quality_score` | 1–10 from the LLM judge (`--skip-quality` to disable) |
| `extras.stats` | per-layer stats from the pipeline |

## Reference results

On the bundled Norwegian sample with gpt-4o-mini extractor and gpt-4o target:

| metric | value |
|---|---|
| tokens_source | 693 |
| tokens_compressed | 392 |
| ratio | **0.57** (43% reduction) |
| cost_baseline | $0.006183 |
| cost_compressed | $0.005370 |
| cost_savings | **13.14%** |
| quality_score | **8/10** |

## Why cost-savings &lt;&lt; token-savings on small inputs

Three reasons:

1. The extract layer still costs ~$0.0005 — eats some of the win.
2. Output narrative tokens are priced the same regardless.
3. A 500-word source is small; input cost is only ~26% of the total.

## Scaling

Savings grow with source size. On a 10k-token source:

- baseline: 10000 × $2.50/1M (input) + 400 × $10/1M (output) ≈ $0.029
- compressed: extract (~$0.001) + 1050 prompt × $2.50/1M + 400 × $10/1M ≈ $0.0077
- **estimated savings: ~74%**

## Honest reporting

When sharing numbers publicly, include:

- the source text (or a representative slice),
- the exact CLI invocation,
- the model versions,
- the timestamp.

Prices change, models drift; a benchmark without context is marketing, not measurement.
