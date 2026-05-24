# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/Mrrobi/narratoflow/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Mrrobi/narratoflow/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/Mrrobi/narratoflow/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/Mrrobi/narratoflow/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/Mrrobi/narratoflow/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Mrrobi/narratoflow/releases/tag/v0.1.0
