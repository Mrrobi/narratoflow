"""Lightweight language detection utilities.

Used when callers pass ``source_lang="auto"`` to the compressor. Detection is
done by the optional ``langdetect`` library if installed; otherwise a small
heuristic that scores text against bundled stopword lists is used.

The heuristic is good enough to distinguish the 12 bundled languages on most
inputs longer than ~100 characters. For short or mixed-language input,
install ``langdetect`` (``pip install 'narratoflow[lang]'``) for better
accuracy.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from pathlib import Path

_SUPPORTED = ("en", "no", "de", "fr", "es", "it", "pt", "nl", "sv", "da", "fi", "pl")


@lru_cache(maxsize=16)
def _load_stopwords(lang: str) -> frozenset[str]:
    fname = f"stopwords_{lang}.txt"
    try:
        with resources.files("narrato.data").joinpath(fname).open("r", encoding="utf-8") as fp:
            return frozenset(w.strip().lower() for w in fp if w.strip())
    except (FileNotFoundError, ModuleNotFoundError):
        local = Path(__file__).parent / "data" / fname
        if local.exists():
            return frozenset(
                w.strip().lower()
                for w in local.read_text(encoding="utf-8").splitlines()
                if w.strip()
            )
        return frozenset()


def detect(text: str, *, default: str = "en") -> str:
    """Detect the ISO 639-1 language code of ``text``.

    Order:

    1. If ``langdetect`` is installed, use it.
    2. Otherwise score against bundled stopword lists.
    3. If text is too short / no signal, return ``default``.
    """
    if not text or len(text.strip()) < 10:
        return default

    try:
        from langdetect import DetectorFactory
        from langdetect import detect as _ld

        DetectorFactory.seed = 0
        code = str(_ld(text)).lower()
        # langdetect returns "no" for Norwegian; "nb"/"nn" are mapped down
        if code in ("nb", "nn"):
            return "no"
        if code in _SUPPORTED:
            return code
    except ImportError:
        pass
    except Exception:
        pass

    return _heuristic(text, default=default)


def _heuristic(text: str, *, default: str) -> str:
    import re

    tokens = [t.lower() for t in re.findall(r"[\wÆØÅæøåÄäÖöÜüÉéÈèÊêÀàÇçÑñÅåÌìÒòÚú]+", text)]
    if not tokens:
        return default

    sample = tokens[:1000]
    sample_set = set(sample)
    sample_size = len(sample)

    best_lang = default
    best_score = 0.0
    for lang in _SUPPORTED:
        stops = _load_stopwords(lang)
        if not stops:
            continue
        hits = sum(1 for tok in sample_set if tok in stops)
        score = hits / sample_size
        if score > best_score:
            best_score = score
            best_lang = lang

    return best_lang if best_score > 0.02 else default


def supported() -> tuple[str, ...]:
    """Return ISO codes that the bundled detector recognises."""
    return _SUPPORTED
