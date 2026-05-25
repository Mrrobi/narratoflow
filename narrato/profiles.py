"""Named compression profiles — opinionated configs on top of the generic core.

A :class:`Profile` is a frozen bundle of ``Compressor`` arguments. Profiles let
users start with one line:

.. code-block:: python

    from narrato import Compressor
    c = Compressor.from_profile("rag-en")

instead of choosing language, schema, layers, chunk size, etc.

Profiles do not change the engine — they are pure convenience wrappers. The
generic core stays language- and domain-neutral; profiles sit on top.

Register your own profiles via :func:`register_profile`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Profile:
    """A named bundle of Compressor defaults."""

    name: str
    description: str
    source_lang: str = "en"
    schema: str = "qa"
    layers: tuple[str, ...] = ("preprocess", "codebook", "extract")
    strip_stopwords: bool = False
    chunked: bool = False
    chunk_chars: int = 8000
    overlap_chars: int = 200
    cache: bool = False
    extra: dict[str, Any] = field(default_factory=dict)
    """Free-form extra kwargs forwarded to ``Compressor()``."""

    def as_compressor_kwargs(self) -> dict[str, Any]:
        """Materialise as a ``Compressor()`` kwargs dict (without provider/model)."""
        kwargs: dict[str, Any] = {
            "source_lang": self.source_lang,
            "schema": self.schema,
            "layers": list(self.layers),
            "chunked": self.chunked,
            "chunk_chars": self.chunk_chars,
            "overlap_chars": self.overlap_chars,
            "cache": self.cache,
        }
        kwargs.update(self.extra)
        return kwargs


# ---------------------------------------------------------------------------
# bundled profiles
# ---------------------------------------------------------------------------


_BUILTINS: list[Profile] = [
    # ---- English (default) ---------------------------------------------------
    Profile(
        name="rag-en",
        description="English RAG / fact extraction. QA schema, stopword stripping on.",
        source_lang="en",
        schema="qa",
        strip_stopwords=True,
    ),
    Profile(
        name="qa-en",
        description="Same as rag-en but stopwords off (preserves grammar for better QA).",
        source_lang="en",
        schema="qa",
    ),
    Profile(
        name="narrative-en",
        description="English narrative generation from source documents.",
        source_lang="en",
        schema="narrative",
    ),
    Profile(
        name="interview-en",
        description="English interview / transcript extraction.",
        source_lang="en",
        schema="interview",
    ),
    Profile(
        name="dialogue-en",
        description="English scripted / fictional dialogue.",
        source_lang="en",
        schema="dialogue",
    ),
    Profile(
        name="news-en",
        description="English news article (5W1H).",
        source_lang="en",
        schema="news",
    ),
    Profile(
        name="long-en",
        description="Long English document — chunked extraction with default QA schema.",
        source_lang="en",
        schema="qa",
        chunked=True,
        chunk_chars=8000,
        overlap_chars=200,
    ),
    # ---- Norwegian ----------------------------------------------------------
    Profile(
        name="narrative-no",
        description="Norwegian narrative generation. Original v0.1–v0.2 defaults.",
        source_lang="no",
        schema="narrative",
    ),
    Profile(
        name="rag-no",
        description="Norwegian RAG / fact extraction.",
        source_lang="no",
        schema="qa",
        strip_stopwords=True,
    ),
]


_REGISTRY: dict[str, Profile] = {p.name: p for p in _BUILTINS}


# ---------------------------------------------------------------------------
# registry API
# ---------------------------------------------------------------------------


def get_profile(name: str) -> Profile:
    """Look up a profile by name. Raises ``KeyError`` if unknown."""
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown profile {name!r}; available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_profiles() -> list[Profile]:
    """Return all registered profiles."""
    return list(_REGISTRY.values())


def register_profile(profile: Profile, *, overwrite: bool = False) -> None:
    """Register a new profile (or overwrite an existing one)."""
    if profile.name in _REGISTRY and not overwrite:
        raise ValueError(
            f"profile {profile.name!r} already registered; pass overwrite=True"
        )
    _REGISTRY[profile.name] = profile


def unregister_profile(name: str) -> None:
    """Remove a profile from the registry."""
    _REGISTRY.pop(name, None)
