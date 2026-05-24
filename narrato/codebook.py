"""Layer 2 — reference compression and codebook substitution.

Builds a per-document codebook of frequent multi-word phrases and assigns each
a short code (e.g. ``§a``, ``§b``). The document is rewritten using the codes,
and the legend is sent alongside so the downstream LLM can decode.

Codes only "win" when the phrase is long enough and frequent enough that the
code + one legend entry costs fewer tokens than the original occurrences.
Because the exact win depends on the tokenizer, we estimate using character
length as a fast proxy and let the pipeline measure real tokens later.

Codes use ``§`` followed by base-36 digits. ``§`` is rare enough in source text
that collisions are unlikely; we strip it from the input first to be safe.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

_CODE_PREFIX = "§"
_WORD = re.compile(r"[\wÆØÅæøå][\wÆØÅæøå'-]*", re.UNICODE)


@dataclass
class CodebookConfig:
    min_phrase_words: int = 2
    max_phrase_words: int = 5
    min_occurrences: int = 3
    min_phrase_chars: int = 8
    max_entries: int = 64
    case_sensitive: bool = False


@dataclass
class CodebookResult:
    text: str
    legend: dict[str, str] = field(default_factory=dict)
    stats: dict = field(default_factory=dict)

    def legend_string(self) -> str:
        """Render legend as compact text block for prepending to prompts."""
        if not self.legend:
            return ""
        lines = [f"{code}={phrase}" for code, phrase in self.legend.items()]
        return "LEGEND:\n" + "\n".join(lines)


def build(text: str, cfg: CodebookConfig | None = None) -> CodebookResult:
    cfg = cfg or CodebookConfig()
    safe_text = text.replace(_CODE_PREFIX, "")
    tokens = list(_WORD.finditer(safe_text))
    if not tokens:
        return CodebookResult(text=safe_text)

    counts: Counter[str] = Counter()
    token_words = [m.group(0) for m in tokens]
    lower_words = [w.lower() for w in token_words] if not cfg.case_sensitive else token_words

    for n in range(cfg.min_phrase_words, cfg.max_phrase_words + 1):
        for i in range(len(lower_words) - n + 1):
            phrase = " ".join(lower_words[i : i + n])
            if len(phrase) < cfg.min_phrase_chars:
                continue
            counts[phrase] += 1

    candidates: list[tuple[str, int, int]] = []
    for phrase, occ in counts.items():
        if occ < cfg.min_occurrences:
            continue
        savings = occ * (len(phrase) - 3) - (len(phrase) + 4)
        if savings <= 0:
            continue
        candidates.append((phrase, occ, savings))

    candidates.sort(key=lambda x: x[2], reverse=True)
    candidates = _drop_subphrases(candidates)
    candidates = candidates[: cfg.max_entries]

    legend: dict[str, str] = {}
    for idx, (phrase, _occ, _savings) in enumerate(candidates):
        legend[_make_code(idx)] = phrase

    rewritten = _apply_legend(safe_text, legend, cfg.case_sensitive)

    return CodebookResult(
        text=rewritten,
        legend=legend,
        stats={
            "candidates_considered": len(counts),
            "entries_emitted": len(legend),
            "chars_in": len(text),
            "chars_out": len(rewritten),
        },
    )


def _drop_subphrases(cands: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    """If phrase A is a substring of higher-savings phrase B with similar count, drop A."""
    keep: list[tuple[str, int, int]] = []
    for phrase, occ, savings in cands:
        dominated = False
        for kept_phrase, kept_occ, _ in keep:
            if phrase in kept_phrase and occ <= kept_occ * 1.1:
                dominated = True
                break
        if not dominated:
            keep.append((phrase, occ, savings))
    return keep


def _make_code(idx: int) -> str:
    digits = "abcdefghijklmnopqrstuvwxyz0123456789"
    if idx < len(digits):
        return f"{_CODE_PREFIX}{digits[idx]}"
    hi = idx // len(digits)
    lo = idx % len(digits)
    return f"{_CODE_PREFIX}{digits[hi]}{digits[lo]}"


def _apply_legend(text: str, legend: dict[str, str], case_sensitive: bool) -> str:
    """Replace phrases in text with codes. Longer phrases first to avoid clobber."""
    items = sorted(legend.items(), key=lambda kv: len(kv[1]), reverse=True)
    flags = 0 if case_sensitive else re.IGNORECASE
    out = text
    for code, phrase in items:
        pattern = re.compile(r"\b" + re.escape(phrase) + r"\b", flags)
        out = pattern.sub(code, out)
    return out


def decode(text: str, legend: dict[str, str]) -> str:
    """Expand codes back to phrases (for testing/debugging round trips)."""
    out = text
    for code, phrase in legend.items():
        out = out.replace(code, phrase)
    return out
