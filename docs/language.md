# Language detection

`narratoflow` picks the right stopword list, spaCy model, and extractor system prompt language based on `Compressor(source_lang=...)`. Pass an ISO 639-1 code (`"en"`, `"no"`, `"de"`, …) or `"auto"` to detect at runtime.

## Auto-detection

```python
from narrato import Compressor

c = Compressor.from_profile("rag-en", provider="anthropic")
c.source_lang = "auto"      # or pass at construction
result = c.compress(text)

print(result.stats["resolved_lang"])
# "en"  -- or "no" / "de" / ... whatever the detector picked
```

## How it works

1. If [`langdetect`](https://pypi.org/project/langdetect/) is installed, use it. Install via the `lang` extra:

   ```bash
   pip install "narratoflow[lang]"
   ```

2. Otherwise, score the input text against the bundled stopword lists (12 languages). The language with the highest stopword hit rate wins.

3. If neither produces a confident signal (text too short, no stopword overlap), fall back to the configured default — usually `"en"`.

## Supported codes

`narrato.language.supported()` returns the codes the bundled heuristic recognises:

```
en  no  de  fr  es  it  pt  nl  sv  da  fi  pl
```

The `langdetect` library recognises ~55 codes; when installed, anything outside the bundled 12 is accepted.

## Programmatic use

```python
from narrato.language import detect

detect("Bonjour le monde, comment ça va?", default="en")
# 'fr'
```

## When NOT to use auto-detect

- Sources you control where the language is fixed. Pass the explicit code; saves a function call per document.
- Very short documents (< 100 chars). The heuristic is noisy below that.
- Mixed-language documents. Detection picks one; the pipeline applies one set of stopwords. Use a profile per language and split the input upstream.
