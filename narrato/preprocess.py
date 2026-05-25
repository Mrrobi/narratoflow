"""Layer 1 — deterministic preprocessing.

No LLM calls. Cheap, language-aware text cleanup that shaves tokens before any
downstream stage runs.

Operations (toggleable):
    - whitespace and punctuation normalization
    - near-duplicate sentence dedupe (Jaccard over token shingles)
    - stopword stripping (optional, lang-aware)

Stopword stripping is OFF by default for the `narrative` preset because narrative
extraction benefits from grammatical context. Turn it on for fact/RAG flows.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[A-ZÆØÅa-zæøå])")
_WS = re.compile(r"\s+")
_PUNCT_RUN = re.compile(r"([.!?,;:])\1+")
_QUOTES = str.maketrans({
    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
    "‘": "'", "’": "'", "‚": "'",
})


def _load_stopwords(lang: str) -> set[str]:
    fname = f"stopwords_{lang}.txt"
    try:
        with resources.files("narrato.data").joinpath(fname).open("r", encoding="utf-8") as fp:
            return {w.strip().lower() for w in fp if w.strip()}
    except (FileNotFoundError, ModuleNotFoundError):
        local = Path(__file__).parent / "data" / fname
        if local.exists():
            return {w.strip().lower() for w in local.read_text(encoding="utf-8").splitlines() if w.strip()}
        return set()


@dataclass
class PreprocessConfig:
    lang: str = "en"
    normalize_unicode: bool = True
    normalize_whitespace: bool = True
    collapse_punct_runs: bool = True
    dedupe_sentences: bool = True
    dedupe_threshold: float = 0.85
    strip_stopwords: bool = False
    min_sentence_chars: int = 4
    spacy_model: str | None = None
    """When set, use spaCy for sentence splitting and POS-aware token stripping.

    May be a full model name (``en_core_web_sm``) or a short ISO code
    (``en`` — maps to the language's default small model). Requires the
    ``nlp`` extra: ``pip install 'narratoflow[nlp]'`` and the model itself
    (``python -m spacy download <model>``). If spaCy is not installed, this
    field is silently ignored and the regex-based default is used.
    """

    spacy_strip_pos: bool = False
    """When True (and ``spacy_model`` is set), drop function-word tokens by POS
    instead of dropping stopwords from the bundled word list. Named entities
    are preserved.
    """


@dataclass
class PreprocessResult:
    text: str
    removed_sentences: int = 0
    stopwords_removed: int = 0
    stats: dict = field(default_factory=dict)


def preprocess(text: str, cfg: PreprocessConfig | None = None) -> PreprocessResult:
    cfg = cfg or PreprocessConfig()
    original_len = len(text)

    if cfg.normalize_unicode:
        text = unicodedata.normalize("NFC", text).translate(_QUOTES)
    if cfg.normalize_whitespace:
        text = _WS.sub(" ", text).strip()
    if cfg.collapse_punct_runs:
        text = _PUNCT_RUN.sub(r"\1", text)

    use_spacy = cfg.spacy_model is not None
    if use_spacy:
        try:
            from narrato.spacy_pipeline import is_available

            use_spacy = is_available()
        except Exception:
            use_spacy = False

    removed_sents = 0
    if cfg.dedupe_sentences:
        if use_spacy:
            text, removed_sents = _dedupe_sentences_spacy(
                text, cfg.dedupe_threshold, cfg.min_sentence_chars, cfg.spacy_model or "en"
            )
        else:
            text, removed_sents = _dedupe_sentences(
                text, cfg.dedupe_threshold, cfg.min_sentence_chars
            )

    stop_removed = 0
    if use_spacy and cfg.spacy_strip_pos:
        from narrato.spacy_pipeline import spacy_strip

        text, stop_removed = spacy_strip(text, model=cfg.spacy_model or "en")
    elif cfg.strip_stopwords:
        text, stop_removed = _strip_stopwords(text, cfg.lang)

    return PreprocessResult(
        text=text,
        removed_sentences=removed_sents,
        stopwords_removed=stop_removed,
        stats={
            "chars_in": original_len,
            "chars_out": len(text),
            "char_ratio": (len(text) / original_len) if original_len else 1.0,
            "spacy_used": use_spacy,
        },
    )


def _dedupe_sentences_spacy(
    text: str, threshold: float, min_chars: int, model: str
) -> tuple[str, int]:
    from narrato.spacy_pipeline import spacy_sentences

    sentences = spacy_sentences(text, model=model)
    kept: list[str] = []
    kept_shingles: list[set[str]] = []
    removed = 0
    for sent in sentences:
        if len(sent) < min_chars:
            kept.append(sent)
            kept_shingles.append(_shingles(sent))
            continue
        sh = _shingles(sent)
        if any(_jaccard(sh, prev) >= threshold for prev in kept_shingles):
            removed += 1
            continue
        kept.append(sent)
        kept_shingles.append(sh)
    return " ".join(kept), removed


def _shingles(s: str, n: int = 3) -> set[str]:
    toks = re.findall(r"\w+", s.lower())
    if len(toks) < n:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _dedupe_sentences(text: str, threshold: float, min_chars: int) -> tuple[str, int]:
    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    kept: list[str] = []
    kept_shingles: list[set[str]] = []
    removed = 0
    for sent in sentences:
        if len(sent) < min_chars:
            kept.append(sent)
            kept_shingles.append(_shingles(sent))
            continue
        sh = _shingles(sent)
        is_dup = any(_jaccard(sh, prev) >= threshold for prev in kept_shingles)
        if is_dup:
            removed += 1
            continue
        kept.append(sent)
        kept_shingles.append(sh)
    return " ".join(kept), removed


def _strip_stopwords(text: str, lang: str) -> tuple[str, int]:
    stop = _load_stopwords(lang)
    if not stop:
        return text, 0
    removed = 0
    out: list[str] = []
    for tok in re.findall(r"\S+", text):
        bare = re.sub(r"[^\wÆØÅæøå']", "", tok).lower()
        if bare and bare in stop:
            removed += 1
            continue
        out.append(tok)
    return " ".join(out), removed
