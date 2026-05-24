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
    lang: str = "no"
    normalize_unicode: bool = True
    normalize_whitespace: bool = True
    collapse_punct_runs: bool = True
    dedupe_sentences: bool = True
    dedupe_threshold: float = 0.85
    strip_stopwords: bool = False
    min_sentence_chars: int = 4


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

    removed_sents = 0
    if cfg.dedupe_sentences:
        text, removed_sents = _dedupe_sentences(text, cfg.dedupe_threshold, cfg.min_sentence_chars)

    stop_removed = 0
    if cfg.strip_stopwords:
        text, stop_removed = _strip_stopwords(text, cfg.lang)

    return PreprocessResult(
        text=text,
        removed_sentences=removed_sents,
        stopwords_removed=stop_removed,
        stats={
            "chars_in": original_len,
            "chars_out": len(text),
            "char_ratio": (len(text) / original_len) if original_len else 1.0,
        },
    )


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
