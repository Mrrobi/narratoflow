# narratoflow

> Compress huge LLM input context into dense intermediate representations. Pay fewer tokens, keep the meaning.

`narratoflow` (PyPI name; import as `narrato`) is an open-source Python library (Apache-2.0) for shrinking long source text before sending it to an LLM. It targets workloads like **narrative generation from large source documents**, where the input dwarfs the output and tokens dominate cost.

## Why

LLM input is priced per token, and a long source document — say a 20-page Norwegian transcript that feeds a 200-word narrative — burns most of the budget *before* the model has written anything.

`narratoflow` trades a tiny bit of fidelity for a large reduction in input tokens by passing your downstream LLM a dense, machine-friendly representation instead of the raw text.

The intermediate representation does not need to be human-readable. It just needs to be:

1. Cheap to produce.
2. Decodable by the downstream LLM into a faithful output.
3. Smaller in tokens than the original.

## Measured

On the bundled Norwegian short-story benchmark (~500 words, gpt-4o-mini extractor → gpt-4o target):

| metric | value |
|---|---|
| input tokens | 693 |
| compressed tokens | 392 |
| **token reduction** | **43%** |
| **cost savings** | **13%** |
| LLM-judge quality | 8/10 |

Cost savings grow sharply with input size — on a 10k-token source, expect ~74% savings.

## Install

```bash
pip install narratoflow
```

Then:

```python
import narrato                       # import name kept stable
from narrato import Compressor, Decoder
```

[Quickstart →](quickstart.md){ .md-button .md-button--primary }
[Architecture →](architecture.md){ .md-button }
