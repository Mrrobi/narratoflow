# API reference

Generated from docstrings via `mkdocstrings`.

## `narrato.pipeline`

::: narrato.pipeline
    options:
      members:
        - Compressor
        - CompressionResult
        - Decoder

## `narrato.preprocess`

::: narrato.preprocess
    options:
      members:
        - preprocess
        - PreprocessConfig
        - PreprocessResult

## `narrato.codebook`

::: narrato.codebook
    options:
      members:
        - build
        - decode
        - CodebookConfig
        - CodebookResult

## `narrato.extractors`

::: narrato.extractors
    options:
      members:
        - extract
        - ExtractResult

## `narrato.schemas`

::: narrato.schemas
    options:
      members:
        - NarrativeFacts
        - QAFacts
        - get_schema
        - schema_to_json_schema

## `narrato.profiles`

::: narrato.profiles
    options:
      members:
        - Profile
        - get_profile
        - list_profiles
        - register_profile
        - unregister_profile

## `narrato.providers`

::: narrato.providers.base
    options:
      members:
        - Provider
        - AsyncProvider
        - ProviderResponse
        - get_provider

::: narrato.providers.ollama
    options:
      members:
        - OllamaProvider

::: narrato.providers.mock
    options:
      members:
        - MockProvider

## `narrato.tokenizers`

::: narrato.tokenizers
    options:
      members:
        - get_tokenizer
        - AnthropicTokenizer
        - OpenAITokenizer

## `narrato.benchmark`

::: narrato.benchmark
    options:
      members:
        - run_benchmark
        - BenchmarkReport
        - judge
        - cost
