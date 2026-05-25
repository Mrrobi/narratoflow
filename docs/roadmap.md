# Roadmap

| version | status | scope |
|---|---|---|
| v0.1 | ✅ shipped | layered preprocess + codebook + schema extract, Anthropic + OpenAI, CLI, eval harness, MkDocs site, auto-publish |
| v0.2 | ✅ shipped | chunked map-reduce extraction, Anthropic prompt caching, 3 new schema presets (interview/dialogue/news), MockProvider, tokenizer-aware codebook |
| v0.3 | ✅ shipped | generic-core refactor: defaults are language-/domain-neutral, opinionated configs live as named profiles. 12-language stopword bundle. CLI gains `--profile`. |
| v0.4 | ✅ shipped | Ollama provider, async API (`acompress`, `extract_async`, `extract_chunked_async`), OpenAI prompt-cache reporting, `py.typed` marker, `uv build` verified, expanded docs site (install, providers, async, typing) |
| v0.5 | ✅ shipped | optional spaCy preprocessing integration, language auto-detect (`source_lang="auto"`), mypy strict-ish config + CI typecheck job |
| v0.6 | planned | learned encoder R&D, HF Spaces demo, multilingual benchmark corpora, more language stopword sets |

## Ideas welcome

Open an issue or PR at [github.com/Mrrobi/narratoflow](https://github.com/Mrrobi/narratoflow).

Concrete asks that would help most:

- Real-world benchmark corpora in non-English languages
- Provider adapters for Google (Gemini), Cohere, Mistral, local Ollama
- Additional schema presets — legal, medical, news, technical-spec
- Quality-judge alternatives (BERTScore, embedding similarity)
