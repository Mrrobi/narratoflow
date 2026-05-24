# Roadmap

| version | status | scope |
|---|---|---|
| v0.1 | ✅ shipped | layered preprocess + codebook + schema extract, Anthropic + OpenAI, CLI, eval harness |
| v0.2 | planned | prompt-cache integration on Anthropic path, more schema presets, HF Spaces demo |
| v0.3 | planned | local model support (Ollama), Norwegian spaCy pipeline integration |
| v0.4 | planned | learned encoder (fine-tuned small model, distributed via Hugging Face) |

## Ideas welcome

Open an issue or PR at [github.com/Mrrobi/narratoflow](https://github.com/Mrrobi/narratoflow).

Concrete asks that would help most:

- Real-world benchmark corpora in non-English languages
- Provider adapters for Google (Gemini), Cohere, Mistral, local Ollama
- Additional schema presets — legal, medical, news, technical-spec
- Quality-judge alternatives (BERTScore, embedding similarity)
