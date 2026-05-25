# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-05-25

### Added
- **Ollama provider** for local models. `narrato.providers.ollama.OllamaProvider`
  talks to a local Ollama daemon over HTTP; reads `OLLAMA_HOST`
  (default `http://localhost:11434`). JSON mode uses Ollama's `format=json`
  plus a schema reminder in the system prompt. Available as `get_provider("ollama")`.
- **Async API** for the whole extraction path:
  - `AsyncProvider` protocol; all built-in providers (Anthropic, OpenAI,
    Ollama, Mock) implement both sync and async surfaces (`acomplete`,
    `acomplete_json`).
  - `Compressor.acompress(text, concurrency=4)` runs the extract layer
    asynchronously. When chunked mode is active, chunks are extracted
    concurrently via `asyncio.gather` bounded by a semaphore.
  - `extract_async()` / `extract_chunked_async()` public functions for direct
    use.
- **OpenAI prompt-cache reporting.** When OpenAI's automatic prompt cache hits
  (prompts ≥ 1024 tokens), `ProviderResponse.cached_input_tokens` is populated
  from `usage.prompt_tokens_details.cached_tokens`. Same field is used by the
  Anthropic adapter for `cache_read_input_tokens`.
- **PEP 561 `py.typed` marker** so mypy/pyright/IDE checkers consume
  `narrato`'s annotations directly. Added `Typing :: Typed` classifier.
- **`httpx` as an explicit dependency** (already transitively present via
  `anthropic` and `openai`; now pinned for Ollama).

### Docs
- New pages: `install.md` (pip + `uv` + from-source + `uv build`),
  `providers.md` (matrix + per-provider details), `async.md` (when and how),
  `typing.md` (type checker setup, custom-schema typing).
- MkDocs nav updated; API reference now covers `narrato.profiles`,
  `OllamaProvider`, `MockProvider`, `AsyncProvider`.

### Build
- Verified `uv build` produces identical wheel/sdist to `python -m build`.
  No pyproject changes needed beyond the existing hatchling backend.

### Tests
- 55 passing (was 42). New `tests/test_async.py`, `tests/test_ollama.py`
  (httpx MockTransport, no network), `tests/test_typing_marker.py`.

## [0.3.0] - 2026-05-25

### Changed (breaking)
- **Generic-core refactor.** Defaults are now language- and domain-neutral.
  - `Compressor(source_lang=...)` defaults from `"no"` → `"en"`
  - `Compressor(schema=...)` defaults from `"narrative"` → `"qa"`
  - These are the only two behaviour-changing defaults; everything else stays.
- Migration: existing users who relied on the old Norwegian-narrative defaults
  can either pass the old arguments explicitly, or switch to a profile:

  ```python
  # v0.2 implicit defaults
  Compressor()                                # was: no + narrative
  # v0.3 equivalent
  Compressor.from_profile("narrative-no")     # explicit, recommended
  Compressor(source_lang="no", schema="narrative")  # also works
  ```

### Added
- `narrato.profiles` module with a `Profile` dataclass and a registry. Nine
  bundled profiles cover the most common starting points
  (`rag-en`, `qa-en`, `narrative-en`, `interview-en`, `dialogue-en`, `news-en`,
  `long-en`, `narrative-no`, `rag-no`).
- `Compressor.from_profile(name, **overrides)` classmethod.
- `register_profile()` / `unregister_profile()` / `list_profiles()` /
  `get_profile()` registry helpers; user-defined profiles are first-class.
- Ten new bundled stopword lists: `de`, `fr`, `es`, `it`, `pt`, `nl`, `sv`,
  `da`, `fi`, `pl` (plus the existing `en` and `no`).
- CLI: `narratoflow profiles` lists all profiles; `narratoflow schemas` lists
  schema presets; `narratoflow compress --profile <name>` starts from a
  profile and accepts any per-call override.

### Why
- The original library framed itself as a "Norwegian narrative" tool, which
  was true to its origin but limited adoption. The engine itself was already
  generic; v0.3 only flips the *defaults* and adds profiles so callers can
  re-opt-in to the previous defaults with one named argument.

## [0.2.0] - 2026-05-25

### Added
- Chunked map-reduce extraction (`narrato.extractors.extract_chunked`) for
  sources too long for a single LLM call. Splits on sentence boundaries with
  configurable overlap; merges partial payloads with order-preserving dedupe.
- `Compressor.chunked`, `Compressor.chunk_chars`, `Compressor.overlap_chars`
  parameters; chunked mode triggers automatically when `len(text) > chunk_chars`.
- Anthropic prompt caching: `Compressor(cache=True)` (or
  `AnthropicProvider(cache=True)`) marks the system prompt as an ephemeral
  cache breakpoint. Repeated calls within ~5 min pay 10 % on the cached portion.
- Three new schema presets: `interview`, `dialogue`, `news` (5W1H).
- `narrato.schemas.list_presets()` helper.
- `narrato.providers.MockProvider` for offline tests and demos. Accepts a
  static payload, a callable, or a sequence of canned responses; counts calls.
- Tokenizer-aware codebook savings: `CodebookConfig.estimator="tokens"` plus
  the `tokenizer` argument to `codebook.build()` switch from char-proxy to
  real-token measurement.
- `Compressor.provider` now accepts a pre-constructed `Provider` instance in
  addition to the string name — useful for dependency injection.

### Changed
- `narrato.tokenizers.get_tokenizer` falls back to the OpenAI/tiktoken
  encoder for unknown provider names instead of raising. Lets mock and custom
  providers participate in pipeline runs without ceremony.
- Codebook layer now wired to the pipeline tokenizer when available; stats
  include the `estimator` used.

### Tests
- New `tests/test_extractors.py`: covers single-shot extract, validation
  errors, chunked map-reduce splitting, overlap dedupe, news schema.
- Pipeline tests now exercise the extract layer end-to-end via MockProvider,
  including chunked mode and the cache flag.

## [0.1.3] - 2026-05-25

### Added
- MkDocs Material documentation site under `docs/` (index, quickstart,
  architecture, schemas, cli, benchmark, API reference, roadmap).
- `.github/workflows/docs.yml` — auto-deploy docs to GitHub Pages.
- `.github/workflows/release.yml` — auto-publish to PyPI via Trusted
  Publishing (OIDC) on `release: published`.
- `docs` extra in `pyproject.toml`.
- Documentation + Changelog URLs in package metadata.

## [0.1.2] - 2026-05-24

### Added
- GitHub Actions CI workflow (`.github/workflows/ci.yml`): pytest matrix on
  Python 3.10/3.11/3.12 across Ubuntu + Windows, separate ruff lint job.

### Changed
- Trimmed README badges to four: PyPI version, Python versions, CI, license.

## [0.1.1] - 2026-05-24

### Added
- README badges (PyPI version, Python versions, license, downloads, GH stars,
  PRs welcome).
- Discoverability tag-chip strip.
- "Highlights" section in README.

## [0.1.0] - 2026-05-24

### Added
- Initial release.
- Layered context compressor: L1 preprocess (deterministic, free), L2
  codebook (deterministic, free), L3 schema-driven semantic extract (cheap
  LLM call).
- Provider adapters for Anthropic (tool-use JSON) and OpenAI
  (`response_format` JSON schema).
- `NarrativeFacts` + `QAFacts` Pydantic schema presets.
- CLI: `narrato compress`, `narrato eval`, `narrato prompt`.
- Round-trip benchmark with LLM judge.
- Norwegian-first preprocessing (stopwords, NFC, sentence dedupe) and
  Norwegian short-story benchmark sample.

[Unreleased]: https://github.com/Mrrobi/narratoflow/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/Mrrobi/narratoflow/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Mrrobi/narratoflow/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Mrrobi/narratoflow/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/Mrrobi/narratoflow/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/Mrrobi/narratoflow/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/Mrrobi/narratoflow/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Mrrobi/narratoflow/releases/tag/v0.1.0
