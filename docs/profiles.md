# Profiles

A **profile** is a named bundle of `Compressor` defaults. Profiles are *pure convenience* — the engine remains generic, language- and domain-neutral. Profiles sit on top so callers can start in one line.

```python
from narrato import Compressor
c = Compressor.from_profile("rag-en")
```

## Built-in profiles

Run `narratoflow profiles` to print the live list. Current bundle:

| name | source_lang | schema | notes |
|---|---|---|---|
| `rag-en` | en | qa | English RAG; stopword stripping on |
| `qa-en` | en | qa | English QA; stopwords preserved for grammar |
| `narrative-en` | en | narrative | English narrative generation |
| `interview-en` | en | interview | English interview / transcript |
| `dialogue-en` | en | dialogue | English scripted dialogue |
| `news-en` | en | news | English news article (5W1H) |
| `long-en` | en | qa | Long English doc — chunked extraction |
| `narrative-no` | no | narrative | Norwegian narrative (original v0.1–v0.2 defaults) |
| `rag-no` | no | qa | Norwegian RAG; stopword stripping on |

## Override profile fields

Any keyword to `from_profile()` overrides the profile's value:

```python
c = Compressor.from_profile(
    "rag-en",
    provider="openai",
    extractor_model="gpt-4o-mini",
    target_model="gpt-4o",
    source_lang="de",   # override the language to German
)
```

## Register your own

```python
from narrato import Profile, register_profile, Compressor

profile = Profile(
    name="legal-en",
    description="English legal documents — chunked extraction with QA schema",
    source_lang="en",
    schema="qa",
    chunked=True,
    chunk_chars=6000,
    overlap_chars=300,
    extra={"cache": True},   # any other Compressor kwargs go in extra
)
register_profile(profile)

c = Compressor.from_profile("legal-en", provider="anthropic")
```

Pass ``overwrite=True`` to `register_profile()` to replace an existing entry.

## Profile or explicit?

- Use a profile when starting a new use case — saves typing and documents intent.
- Use explicit construction when the profile fields would need so many overrides that the profile no longer reads cleanly.
- Define your own profile if your team will run the same configuration repeatedly.
