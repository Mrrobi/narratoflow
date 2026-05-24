"""Layer 2 — reference compression and codebook substitution.

Builds a per-document codebook of frequent multi-word phrases and assigns each
a short code (e.g. ``§a``, ``§b``). The document is rewritten using the codes,
and the legend is sent alongside so the downstream LLM can decode.

Codes only "win" when the phrase is long enough and frequent enough that the
code + one legend entry costs fewer tokens than the original occurrences.

Two savings estimators are supported:
    1. ``"chars"`` (default, free) — uses character length as a fast proxy.
    2. ``"tokens"`` — measures real token counts via a :class:`Tokenizer`. More
       accurate, costs a handful of tokenizer calls per candidate. Pass a
       tokenizer to :func:`build` to enable.

Codes use ``§`` followed by base-36 digits. ``§`` is rare enough in source text
that collisions are unlikely; we strip it from the input first to be safe.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

_CODE_PREFIX = "§"
_WORD = re.compile(r"[\wÆØÅæøå][\wÆØÅæøå'-]*", re.UNICODE)


class _CountableTokenizer(Protocol):
    def count(self, text: str) -> int:
        ...


@dataclass
class CodebookConfig:
    min_phrase_words: int = 2
    max_phrase_words: int = 5
    min_occurrences: int = 3
    min_phrase_chars: int = 8
    max_entries: int = 64
    case_sensitive: bool = False
    estimator: str = "chars"
    """Either ``"chars"`` (default, free) or ``"tokens"`` (real-token measurement)."""


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


def build(
    text: str,
    cfg: CodebookConfig | None = None,
    *,
    tokenizer: _CountableTokenizer | None = None,
) -> CodebookResult:
    """Build a codebook from ``text``.

    Parameters
    ----------
    text:
        Source text.
    cfg:
        Codebook configuration (thresholds, max entries, estimator choice).
    tokenizer:
        Optional tokenizer. When provided and ``cfg.estimator == "tokens"``,
        savings are computed using real token counts; otherwise char-length is
        used as a fast proxy.
    """
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

    use_tokens = cfg.estimator == "tokens" and tokenizer is not None

    candidates: list[tuple[str, int, int]] = []
    for phrase, occ in counts.items():
        if occ < cfg.min_occurrences:
            continue
        savings = _estimate_savings(phrase, occ, tokenizer=tokenizer if use_tokens else None)
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
            "estimator": "tokens" if use_tokens else "chars",
        },
    )


def _estimate_savings(
    phrase: str, occ: int, *, tokenizer: _CountableTokenizer | None
) -> int:
    """Estimated *token* savings if ``phrase`` is replaced by a 2-char code.

    When a tokenizer is provided we measure real tokens; otherwise we use
    character length as a coarse proxy. The legend cost is `phrase_size + 4`
    (the legend line ``§x=phrase\\n`` plus a leading space/newline).
    """
    if tokenizer is not None:
        phrase_size = tokenizer.count(phrase)
        code_size = 1   # one BPE token for typical short ``§x`` codes
        legend_cost = phrase_size + 3
    else:
        phrase_size = len(phrase)
        code_size = 3   # 3 chars for ``§x ``
        legend_cost = phrase_size + 4
    return occ * (phrase_size - code_size) - legend_cost


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
