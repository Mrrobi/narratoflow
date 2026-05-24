# CLI

The package installs two console scripts: `narratoflow` and the shorter alias `narrato`. Both expose the same commands.

## `compress`

Compress a text file and emit a JSON envelope.

```bash
narratoflow compress input.txt \
  --lang no \
  --provider anthropic \
  --extractor-model claude-haiku-4-5-20251001 \
  --target-model claude-opus-4-7 \
  --schema narrative \
  --out compressed.json
```

| flag | default | meaning |
|---|---|---|
| `--lang` | `no` | source language ISO code |
| `--provider` | `anthropic` | `anthropic` or `openai` |
| `--extractor-model` | `claude-haiku-4-5-20251001` | cheap model used for L3 |
| `--target-model` | `claude-opus-4-7` | model token-counter calibrates against |
| `--schema` | `narrative` | preset name or import path |
| `--layer` | (all) | repeat to pick a subset, e.g. `--layer preprocess --layer codebook` |
| `--out` | (stdout) | output file |

## `eval`

End-to-end benchmark: tokens, cost savings, LLM-judge quality.

```bash
narratoflow eval input.txt \
  --target-task "Skriv en kort fortelling (200 ord) på norsk." \
  --provider openai \
  --extractor-model gpt-4o-mini \
  --target-model gpt-4o
```

Add `--skip-quality` for token-only runs (cheaper).

## `prompt`

Build a ready-to-send prompt from an existing `compress` JSON output.

```bash
narratoflow prompt compressed.json \
  --instruction "Skriv en kort fortelling basert på fakta." \
  --target-lang no
```

## Environment

Provider keys are read from:

- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`

If you keep them in `.env`, source it in your shell or load it via `python-dotenv` in your own wrapper.
