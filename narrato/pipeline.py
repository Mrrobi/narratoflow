"""High-level orchestrator: ``Compressor`` chains layers and returns a
``CompressionResult``. ``Decoder`` builds a ready-to-send prompt from the
result.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from pydantic import BaseModel

from narrato.codebook import CodebookConfig
from narrato.codebook import build as codebook_build
from narrato.extractors import (
    extract,
    extract_async,
    extract_chunked,
    extract_chunked_async,
)
from narrato.preprocess import PreprocessConfig, preprocess
from narrato.providers.base import AsyncProvider, Provider, get_provider
from narrato.schemas import get_schema
from narrato.tokenizers import Tokenizer, get_tokenizer

_DEFAULT_LAYERS = ("preprocess", "codebook", "extract")
_VALID_LAYERS = {"preprocess", "codebook", "extract"}


@dataclass
class CompressionResult:
    payload: str
    """The compressed payload, ready to be embedded in a downstream prompt.

    For ``extract`` runs this is a JSON object as a string. For runs without
    ``extract`` this is the rewritten text.
    """

    legend: dict[str, str] = field(default_factory=dict)
    """Codebook legend; empty when codebook layer is not used."""

    format: str = "text"
    """``"text"`` or ``"json"``."""

    layers_run: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "payload": self.payload,
            "legend": self.legend,
            "format": self.format,
            "layers_run": self.layers_run,
            "stats": self.stats,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


@dataclass
class Compressor:
    """End-to-end context compressor.

    Parameters
    ----------
    source_lang:
        ISO code for the source text language. Drives stopword selection.
    provider:
        ``"anthropic"`` or ``"openai"`` — used for the extractor LLM call. May
        also be a pre-constructed provider instance (useful for testing /
        injecting :class:`narrato.providers.MockProvider`).
    extractor_model:
        Cheap model used for L3 extraction. e.g. ``claude-haiku-4-5-20251001``
        or ``gpt-4o-mini``.
    target_model:
        Model the downstream prompt will be sent to. Used only for token
        accounting in stats.
    layers:
        Subset of ``("preprocess", "codebook", "extract")`` to run, in order.
    schema:
        Preset name or Pydantic model class. Only used if ``extract`` is in
        ``layers``.
    preprocess_config / codebook_config:
        Override default configs for L1 / L2.
    cache:
        Enable provider-side prompt caching when supported (Anthropic only as
        of v0.2). Reduces cost on repeated calls within the cache TTL.
    chunked:
        Split very long source text into chunks for the extract layer and
        map-reduce the results. Triggers automatically when ``len(text) >
        chunk_chars`` even if this flag is False — set explicitly to force
        chunked mode for testing.
    chunk_chars / overlap_chars:
        Chunk size and overlap for chunked extraction.
    """

    source_lang: str = "en"
    provider: str | Provider = "anthropic"
    extractor_model: str = "claude-haiku-4-5-20251001"
    target_model: str = "claude-opus-4-7"
    layers: tuple[str, ...] | list[str] = _DEFAULT_LAYERS
    schema: str | type[BaseModel] = "qa"
    preprocess_config: PreprocessConfig | None = None
    codebook_config: CodebookConfig | None = None
    cache: bool = False
    chunked: bool = False
    chunk_chars: int = 8000
    overlap_chars: int = 200
    max_tokens: int = 4096
    temperature: float = 0.0
    _provider_obj: Provider | None = None
    _tokenizer: Tokenizer | None = None

    def __post_init__(self) -> None:
        for layer in self.layers:
            if layer not in _VALID_LAYERS:
                raise ValueError(f"unknown layer {layer!r}; valid: {sorted(_VALID_LAYERS)}")
        if not isinstance(self.provider, str):
            self._provider_obj = self.provider

    @classmethod
    def from_profile(
        cls,
        profile: str,
        *,
        provider: str | Provider | None = None,
        extractor_model: str | None = None,
        target_model: str | None = None,
        **overrides: Any,
    ) -> Compressor:
        """Construct a Compressor from a named profile.

        Profile fields become defaults; ``provider``, ``extractor_model``,
        ``target_model`` and any keyword ``overrides`` are applied on top.

        >>> Compressor.from_profile("rag-en", provider="openai")
        """
        from narrato.profiles import get_profile

        prof = get_profile(profile)
        kwargs = prof.as_compressor_kwargs()
        if provider is not None:
            kwargs["provider"] = provider
        if extractor_model is not None:
            kwargs["extractor_model"] = extractor_model
        if target_model is not None:
            kwargs["target_model"] = target_model
        kwargs.update(overrides)

        if prof.strip_stopwords and "preprocess_config" not in kwargs:
            kwargs["preprocess_config"] = PreprocessConfig(
                lang=kwargs.get("source_lang", prof.source_lang),
                strip_stopwords=True,
            )
        return cls(**kwargs)

    def _get_provider(self) -> Provider:
        if self._provider_obj is None:
            assert isinstance(self.provider, str)
            self._provider_obj = get_provider(self.provider, cache=self.cache)
        return self._provider_obj

    @property
    def provider_name(self) -> str:
        if isinstance(self.provider, str):
            return self.provider
        return getattr(self.provider, "name", "custom")

    def _get_tokenizer(self) -> Tokenizer:
        if self._tokenizer is None:
            self._tokenizer = get_tokenizer(self.provider_name, self.target_model)
        return self._tokenizer

    def compress(self, text: str) -> CompressionResult:
        tok = self._get_tokenizer()
        tokens_in = tok.count(text)
        layer_stats: dict[str, Any] = {"input_tokens": tokens_in}
        ran: list[str] = []

        current = text
        legend: dict[str, str] = {}
        fmt = "text"

        if "preprocess" in self.layers:
            res = preprocess(
                current,
                self.preprocess_config or PreprocessConfig(lang=self.source_lang),
            )
            current = res.text
            layer_stats["preprocess"] = {
                **res.stats,
                "removed_sentences": res.removed_sentences,
                "stopwords_removed": res.stopwords_removed,
                "tokens_after": tok.count(current),
            }
            ran.append("preprocess")

        if "codebook" in self.layers:
            res = codebook_build(
                current,
                self.codebook_config or CodebookConfig(),
                tokenizer=tok,
            )
            current = res.text
            legend = res.legend
            layer_stats["codebook"] = {
                **res.stats,
                "tokens_after": tok.count(current),
                "tokens_legend": tok.count(res.legend_string()) if res.legend else 0,
            }
            ran.append("codebook")

        if "extract" in self.layers:
            schema_cls = get_schema(self.schema)
            use_chunked = self.chunked or len(current) > self.chunk_chars

            if use_chunked:
                res = extract_chunked(
                    current,
                    schema=schema_cls,
                    provider=self._get_provider(),
                    model=self.extractor_model,
                    chunk_chars=self.chunk_chars,
                    overlap_chars=self.overlap_chars,
                    legend=legend or None,
                    source_lang=self.source_lang,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                current = json.dumps(res.payload, ensure_ascii=False)
                fmt = "json"
                legend = {}
                layer_stats["extract"] = {
                    "mode": "chunked",
                    "chunks": res.chunks,
                    "model": res.model,
                    "input_tokens": res.input_tokens,
                    "output_tokens": res.output_tokens,
                    "valid": res.valid,
                    "validation_error": res.validation_error,
                    "tokens_after": tok.count(current),
                }
            else:
                single = extract(
                    current,
                    schema=schema_cls,
                    provider=self._get_provider(),
                    model=self.extractor_model,
                    legend=legend or None,
                    source_lang=self.source_lang,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                current = json.dumps(single.payload, ensure_ascii=False)
                fmt = "json"
                legend = {}
                layer_stats["extract"] = {
                    "mode": "single",
                    "model": single.model,
                    "input_tokens": single.input_tokens,
                    "output_tokens": single.output_tokens,
                    "valid": single.valid,
                    "validation_error": single.validation_error,
                    "tokens_after": tok.count(current),
                }
            ran.append("extract")

        tokens_out = tok.count(current)
        layer_stats["output_tokens"] = tokens_out
        layer_stats["ratio"] = (tokens_out / tokens_in) if tokens_in else 1.0
        layer_stats["target_model"] = self.target_model
        layer_stats["provider"] = self.provider_name
        layer_stats["cache_enabled"] = self.cache

        return CompressionResult(
            payload=current,
            legend=legend,
            format=fmt,
            layers_run=ran,
            stats=layer_stats,
        )

    async def acompress(self, text: str, *, concurrency: int = 4) -> CompressionResult:
        """Async equivalent of :meth:`compress`.

        Free layers run synchronously (they are pure CPU and fast); the extract
        layer uses the provider's async methods. When chunked mode is active
        the chunks are extracted concurrently with the given ``concurrency``.
        Requires the configured provider to implement
        :class:`narrato.providers.AsyncProvider`.
        """
        tok = self._get_tokenizer()
        tokens_in = tok.count(text)
        layer_stats: dict[str, Any] = {"input_tokens": tokens_in}
        ran: list[str] = []

        current = text
        legend: dict[str, str] = {}
        fmt = "text"

        if "preprocess" in self.layers:
            res = preprocess(
                current,
                self.preprocess_config or PreprocessConfig(lang=self.source_lang),
            )
            current = res.text
            layer_stats["preprocess"] = {
                **res.stats,
                "removed_sentences": res.removed_sentences,
                "stopwords_removed": res.stopwords_removed,
                "tokens_after": tok.count(current),
            }
            ran.append("preprocess")

        if "codebook" in self.layers:
            cb_res = codebook_build(
                current,
                self.codebook_config or CodebookConfig(),
                tokenizer=tok,
            )
            current = cb_res.text
            legend = cb_res.legend
            layer_stats["codebook"] = {
                **cb_res.stats,
                "tokens_after": tok.count(current),
                "tokens_legend": tok.count(cb_res.legend_string()) if cb_res.legend else 0,
            }
            ran.append("codebook")

        if "extract" in self.layers:
            schema_cls = get_schema(self.schema)
            use_chunked = self.chunked or len(current) > self.chunk_chars
            aprovider: AsyncProvider = self._get_provider()  # type: ignore[assignment]

            if use_chunked:
                ch_res = await extract_chunked_async(
                    current,
                    schema=schema_cls,
                    provider=aprovider,
                    model=self.extractor_model,
                    chunk_chars=self.chunk_chars,
                    overlap_chars=self.overlap_chars,
                    legend=legend or None,
                    source_lang=self.source_lang,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    concurrency=concurrency,
                )
                current = json.dumps(ch_res.payload, ensure_ascii=False)
                fmt = "json"
                legend = {}
                layer_stats["extract"] = {
                    "mode": "chunked-async",
                    "chunks": ch_res.chunks,
                    "concurrency": concurrency,
                    "model": ch_res.model,
                    "input_tokens": ch_res.input_tokens,
                    "output_tokens": ch_res.output_tokens,
                    "valid": ch_res.valid,
                    "validation_error": ch_res.validation_error,
                    "tokens_after": tok.count(current),
                }
            else:
                single = await extract_async(
                    current,
                    schema=schema_cls,
                    provider=aprovider,
                    model=self.extractor_model,
                    legend=legend or None,
                    source_lang=self.source_lang,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                current = json.dumps(single.payload, ensure_ascii=False)
                fmt = "json"
                legend = {}
                layer_stats["extract"] = {
                    "mode": "single-async",
                    "model": single.model,
                    "input_tokens": single.input_tokens,
                    "output_tokens": single.output_tokens,
                    "valid": single.valid,
                    "validation_error": single.validation_error,
                    "tokens_after": tok.count(current),
                }
            ran.append("extract")

        tokens_out = tok.count(current)
        layer_stats["output_tokens"] = tokens_out
        layer_stats["ratio"] = (tokens_out / tokens_in) if tokens_in else 1.0
        layer_stats["target_model"] = self.target_model
        layer_stats["provider"] = self.provider_name
        layer_stats["cache_enabled"] = self.cache

        return CompressionResult(
            payload=current,
            legend=legend,
            format=fmt,
            layers_run=ran,
            stats=layer_stats,
        )


class Decoder:
    """Builds the final prompt to send to the downstream LLM."""

    @staticmethod
    def unpack_prompt(
        result: CompressionResult,
        instruction: str,
        *,
        target_lang: str | None = None,
    ) -> str:
        """Return a single-string prompt embedding payload, legend, and instruction.

        For Anthropic, this is suitable as the ``user`` message. The caller is
        free to split the legend onto a cached system prompt instead — see
        ``unpack_messages``.
        """
        parts: list[str] = []
        if result.legend:
            parts.append("LEGEND (code = phrase):")
            parts.extend(f"  {c}={p}" for c, p in result.legend.items())
            parts.append("")
        if result.format == "json":
            parts.append("FACTS (JSON):")
            parts.append(result.payload)
        else:
            parts.append("SOURCE (may contain legend codes):")
            parts.append(result.payload)
        parts.append("")
        parts.append("INSTRUCTION:")
        if target_lang:
            parts.append(f"(Write the output in language: {target_lang})")
        parts.append(instruction)
        return "\n".join(parts)

    @staticmethod
    def unpack_messages(
        result: CompressionResult,
        instruction: str,
        *,
        target_lang: str | None = None,
    ) -> dict[str, Any]:
        """Return a ``{system, user}`` split so the legend can be cached server-side."""
        system_parts: list[str] = []
        if result.legend:
            system_parts.append("LEGEND (code = phrase):")
            system_parts.extend(f"  {c}={p}" for c, p in result.legend.items())
        system = "\n".join(system_parts) if system_parts else ""

        user_parts: list[str] = []
        if result.format == "json":
            user_parts.append("FACTS (JSON):")
            user_parts.append(result.payload)
        else:
            user_parts.append("SOURCE:")
            user_parts.append(result.payload)
        user_parts.append("")
        if target_lang:
            user_parts.append(f"(Write the output in language: {target_lang})")
        user_parts.append(instruction)

        return {"system": system, "user": "\n".join(user_parts)}


def to_serializable(obj: Any) -> Any:
    """Recursively convert dataclasses/Pydantic models to plain dicts."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [to_serializable(v) for v in obj]
    return obj
