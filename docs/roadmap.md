# Roadmap

| version | status | scope |
|---|---|---|
| v0.1 | ✅ shipped | layered preprocess + codebook + schema extract, Anthropic + OpenAI, CLI, eval harness, MkDocs site, auto-publish |
| v0.2 | ✅ shipped | chunked map-reduce extraction, Anthropic prompt caching, 3 new schema presets (interview/dialogue/news), MockProvider, tokenizer-aware codebook |
| v0.3 | ✅ shipped | generic-core refactor: defaults are language-/domain-neutral, opinionated configs live as named profiles. 12-language stopword bundle. CLI gains `--profile`. |
| v0.4 | planned | local model support (Ollama), spaCy pipeline integration, async API, OpenAI prompt caching when available |
| v0.5 | planned | learned encoder (fine-tuned small model, distributed via Hugging Face) |
| v0.6 | planned | HF Spaces demo, multilingual benchmark corpora, more language stopword sets |

## Ideas welcome

Open an issue or PR at [github.com/Mrrobi/narratoflow](https://github.com/Mrrobi/narratoflow).

Concrete asks that would help most:

- Real-world benchmark corpora in non-English languages
- Provider adapters for Google (Gemini), Cohere, Mistral, local Ollama
- Additional schema presets — legal, medical, news, technical-spec
- Quality-judge alternatives (BERTScore, embedding similarity)
