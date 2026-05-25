# spaCy integration

`narratoflow` ships an **optional** spaCy integration. spaCy is not a hard dependency — install it via the `nlp` extra when you need higher-quality preprocessing.

```bash
pip install "narratoflow[nlp]"
python -m spacy download en_core_web_sm    # or your language's model
```

## When to use spaCy

| use case | use spaCy? |
|---|---|
| English-only source, short docs | optional |
| Mixed-language or non-English | recommended |
| Languages where regex sentence splitting struggles (German compounds, Finnish, Polish) | recommended |
| POS-aware token stripping (drop function words while keeping entities) | required |
| Existing regex fallback works fine | skip the dep |

## Enable on a Compressor

Pass `spacy_model` on a `PreprocessConfig`:

```python
from narrato import Compressor
from narrato.preprocess import PreprocessConfig

c = Compressor.from_profile(
    "rag-en",
    provider="anthropic",
    preprocess_config=PreprocessConfig(
        lang="en",
        spacy_model="en_core_web_sm",   # full model name
    ),
)
```

You can also pass a short ISO code (`"en"`, `"no"`, `"de"`, …) — narrato resolves it to the language's default small model via `narrato.spacy_pipeline.model_for_lang`.

## POS-aware token stripping

Replace stopword-list stripping with spaCy POS filtering. Named entities are preserved automatically.

```python
PreprocessConfig(
    spacy_model="en_core_web_sm",
    spacy_strip_pos=True,
)
```

Default dropped POS: `ADP`, `AUX`, `CCONJ`, `DET`, `PART`, `PRON`, `SCONJ`. Override by editing `narrato.spacy_pipeline.DEFAULT_DROP_POS` or by calling `spacy_strip()` directly.

## Direct use

```python
from narrato.spacy_pipeline import spacy_sentences, spacy_strip

sents = spacy_sentences("First sentence. Second one.", model="en_core_web_sm")

stripped, dropped = spacy_strip(
    "The quick brown fox jumps over the lazy dog.",
    model="en_core_web_sm",
    keep_entities=True,
)
print(stripped, "dropped tokens:", dropped)
```

## Graceful fallback

If spaCy is not installed, the `spacy_model` field is silently ignored and the regex-based default runs. You will see `spacy_used: False` in the stats. No error, no crash — your pipeline keeps working.

## Performance

spaCy adds ~50-200 ms cold start (model load) and is roughly 10× slower than the regex fallback per document. The first call within a process pays the model-load cost; subsequent calls reuse a cached `Language` object via `functools.lru_cache`.
