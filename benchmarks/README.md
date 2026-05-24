# benchmarks

Run reproducible benchmarks to measure compression ratio and downstream narrative quality.

## Sample data

- `samples/norwegian_sample.txt` — Norwegian short story, ~500 words, used as the canonical Norwegian-narrative benchmark.

Add your own samples in `samples/`; the file name (without extension) becomes the benchmark ID.

## Quick run

```bash
narrato eval benchmarks/samples/norwegian_sample.txt \
  --target-task "Skriv en kort fortelling (200-300 ord) på norsk basert på fakta." \
  --provider anthropic \
  --extractor-model claude-haiku-4-5-20251001 \
  --target-model claude-opus-4-7 \
  --schema narrative
```

Skip the quality judge (and the two narrative generations it requires) for a token-only run:

```bash
narrato eval benchmarks/samples/norwegian_sample.txt \
  --target-task "Skriv en kort fortelling." \
  --skip-quality
```

## Reading the report

The `eval` command emits a JSON object with:

| field | meaning |
|------:|---------|
| `tokens_source` | tokens in the raw source, measured for the target model |
| `tokens_compressed` | tokens in the final downstream prompt (legend + payload + instruction) |
| `ratio` | `tokens_compressed / tokens_source` — lower is better |
| `cost_baseline` | estimated USD cost for the baseline narrative generation (full source) |
| `cost_compressed` | estimated USD cost for the compressed run (incl. extractor call) |
| `cost_savings_pct` | percent saved versus baseline |
| `quality_score` | 1-10 from the LLM judge (skipped if `--skip-quality`) |
| `extras.stats` | per-layer stats from the pipeline |

## Honest reporting

When sharing numbers publicly, include:

- the source text,
- the exact CLI invocation,
- the model versions,
- and the timestamp.

Prices change, models drift; a benchmark without context is marketing, not measurement.
