"""Optional spaCy integration for higher-quality preprocessing.

This module is imported lazily by :mod:`narrato.preprocess`. spaCy is *not*
a hard dependency of `narratoflow`; install it via the ``nlp`` extra:

.. code-block:: bash

    pip install "narratoflow[nlp]"
    python -m spacy download en_core_web_sm   # or your language model

When :class:`~narrato.preprocess.PreprocessConfig` has ``spacy_model`` set and
spaCy is importable, the preprocess layer uses spaCy for sentence splitting
and (optionally) for stripping non-content tokens by POS or by lemma. If
spaCy is not installed, the preprocess layer falls back to the regex-based
default with no error.

Public functions:

* :func:`load_model` — cached loader; raises a clear error when missing.
* :func:`spacy_sentences` — sentence-split using spaCy.
* :func:`spacy_strip` — drop tokens that match a POS filter, preserving order.

Performance: spaCy is heavier than the regex fallback (~10× slower on small
docs). It pays off on longer documents and in languages where naïve regex
sentence splitting breaks down (German compound words, Finnish, etc.).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


# Map ISO language codes to common spaCy model names. Users can override by
# passing the full model name (e.g. ``de_core_news_lg``) to PreprocessConfig.
DEFAULT_MODELS: dict[str, str] = {
    "en": "en_core_web_sm",
    "de": "de_core_news_sm",
    "fr": "fr_core_news_sm",
    "es": "es_core_news_sm",
    "it": "it_core_news_sm",
    "pt": "pt_core_news_sm",
    "nl": "nl_core_news_sm",
    "sv": "sv_core_news_sm",
    "da": "da_core_news_sm",
    "fi": "fi_core_news_sm",
    "pl": "pl_core_news_sm",
    "no": "nb_core_news_sm",
}

# POS tags that are typically safe to drop for compression. Override per-call.
DEFAULT_DROP_POS: frozenset[str] = frozenset(
    {"ADP", "AUX", "CCONJ", "DET", "PART", "PRON", "SCONJ"}
)


def model_for_lang(lang: str) -> str:
    """Best-effort mapping from ISO code to a default small spaCy model."""
    return DEFAULT_MODELS.get(lang.lower(), "en_core_web_sm")


@lru_cache(maxsize=8)
def load_model(name_or_lang: str) -> Any:
    """Load and cache a spaCy ``Language`` object.

    ``name_or_lang`` may be either a full model name (``en_core_web_sm``) or a
    short ISO code (``en``, ``no``). Raises a :class:`RuntimeError` with a
    helpful message if spaCy or the model is not installed.
    """
    try:
        import spacy
    except ImportError as e:  # pragma: no cover - exercised only without dep
        raise RuntimeError(
            "spaCy is not installed. Install with: pip install 'narratoflow[nlp]'"
        ) from e

    name = name_or_lang if "_" in name_or_lang else model_for_lang(name_or_lang)
    try:
        return spacy.load(name, disable=["ner"])
    except OSError as e:  # pragma: no cover - depends on installed models
        raise RuntimeError(
            f"spaCy model {name!r} not found. Install with: "
            f"python -m spacy download {name}"
        ) from e


def spacy_sentences(text: str, *, model: str) -> list[str]:
    """Sentence-split ``text`` using spaCy. Falls through with empty list on empty input."""
    if not text:
        return []
    nlp = load_model(model)
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def spacy_strip(
    text: str,
    *,
    model: str,
    drop_pos: frozenset[str] | set[str] | None = None,
    keep_entities: bool = True,
) -> tuple[str, int]:
    """Strip function-word tokens (by POS) while preserving content.

    Returns the rewritten text and the count of dropped tokens. Named entities
    are kept verbatim regardless of POS when ``keep_entities`` is true.
    """
    if not text:
        return text, 0
    nlp = load_model(model)
    doc = nlp(text)
    drop = set(drop_pos) if drop_pos is not None else set(DEFAULT_DROP_POS)
    ent_token_ids: set[int] = set()
    if keep_entities:
        for ent in doc.ents:
            for tok in ent:
                ent_token_ids.add(tok.i)

    kept: list[str] = []
    dropped = 0
    for tok in doc:
        if tok.is_space:
            continue
        if tok.is_punct:
            kept.append(tok.text)
            continue
        if tok.i in ent_token_ids:
            kept.append(tok.text)
            continue
        if tok.pos_ in drop:
            dropped += 1
            continue
        kept.append(tok.text)

    # Reassemble with a single space; punctuation re-attach is approximate.
    out = " ".join(kept)
    out = out.replace(" ,", ",").replace(" .", ".").replace(" ;", ";").replace(" !", "!").replace(" ?", "?")
    return out, dropped


def is_available() -> bool:
    """Return True if spaCy is installed and importable."""
    try:
        import spacy  # noqa: F401
        return True
    except ImportError:
        return False
