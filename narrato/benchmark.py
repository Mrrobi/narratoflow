"""Round-trip benchmark: measure tokens-in, tokens-out, and downstream quality.

Quality is scored by an LLM judge that compares two narratives generated from
(a) the full source vs (b) the compressed payload. A score from 1-10 is
returned along with a short rationale.

Cost numbers are *estimates* using the price table below; update as providers
change pricing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from narrato.pipeline import CompressionResult, Compressor, Decoder
from narrato.providers.base import Provider, get_provider
from narrato.tokenizers import get_tokenizer

# USD per 1M tokens. Keep this list narrow and update via PR when prices change.
PRICES_PER_M: dict[str, tuple[float, float]] = {
    # anthropic
    "claude-opus-4-7": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    # openai (illustrative — verify before billing)
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
}


def cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p = PRICES_PER_M.get(model)
    if not p:
        return 0.0
    inp, out = p
    return (input_tokens / 1_000_000) * inp + (output_tokens / 1_000_000) * out


_JUDGE_SYSTEM = (
    "You are an impartial judge comparing two narratives generated from the same "
    "source material. Score how well narrative B matches the factual content, tone, "
    "and important details of narrative A (treated as ground truth). Score 1-10. "
    "Output strict JSON: {\"score\": int, \"rationale\": str}."
)


def judge(
    reference_narrative: str,
    candidate_narrative: str,
    *,
    provider: Provider,
    model: str,
) -> dict:
    user = (
        f"NARRATIVE A (reference):\n{reference_narrative}\n\n"
        f"NARRATIVE B (candidate):\n{candidate_narrative}\n\n"
        "Score B against A. Output JSON only."
    )
    resp = provider.complete_json(
        system=_JUDGE_SYSTEM,
        user=user,
        model=model,
        max_tokens=512,
        temperature=0.0,
    )
    try:
        return json.loads(resp.text)
    except Exception:
        return {"score": None, "rationale": resp.text}


@dataclass
class BenchmarkReport:
    tokens_source: int
    tokens_compressed: int
    ratio: float
    cost_baseline: float
    cost_compressed: float
    cost_savings_pct: float
    quality_score: int | float | None
    quality_rationale: str | None
    layers_run: list[str]
    extras: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False, indent=2)


def run_benchmark(
    source_text: str,
    *,
    instruction: str,
    compressor: Compressor,
    target_model: str | None = None,
    judge_model: str | None = None,
    skip_quality: bool = False,
) -> BenchmarkReport:
    """Run an end-to-end benchmark on one document."""

    target_model = target_model or compressor.target_model
    provider = get_provider(compressor.provider)

    result: CompressionResult = compressor.compress(source_text)
    tok = get_tokenizer(compressor.provider, target_model)
    tokens_source = tok.count(source_text)
    tokens_compressed = tok.count(Decoder.unpack_prompt(result, instruction=instruction))

    ratio = (tokens_compressed / tokens_source) if tokens_source else 1.0

    extract_cost = 0.0
    extract = result.stats.get("extract")
    if isinstance(extract, dict):
        extract_cost = cost(
            extract.get("model", compressor.extractor_model),
            extract.get("input_tokens", 0),
            extract.get("output_tokens", 0),
        )

    quality_score: int | float | None = None
    quality_rationale: str | None = None
    baseline_out_tokens = 0
    compressed_out_tokens = 0

    if not skip_quality:
        baseline_resp = provider.complete(
            system="Write a faithful short narrative based on the source.",
            user=f"SOURCE:\n{source_text}\n\nTASK:\n{instruction}",
            model=target_model,
            max_tokens=1024,
            temperature=0.2,
        )
        baseline_out_tokens = baseline_resp.output_tokens

        compressed_resp = provider.complete(
            system="Write a faithful short narrative based on the facts.",
            user=Decoder.unpack_prompt(result, instruction=instruction),
            model=target_model,
            max_tokens=1024,
            temperature=0.2,
        )
        compressed_out_tokens = compressed_resp.output_tokens

        judged = judge(
            reference_narrative=baseline_resp.text,
            candidate_narrative=compressed_resp.text,
            provider=provider,
            model=judge_model or compressor.extractor_model,
        )
        quality_score = judged.get("score")
        quality_rationale = judged.get("rationale")

    baseline_cost = cost(target_model, tokens_source, baseline_out_tokens or 400)
    compressed_total = cost(target_model, tokens_compressed, compressed_out_tokens or 400) + extract_cost
    savings_pct = (1 - compressed_total / baseline_cost) * 100 if baseline_cost else 0.0

    return BenchmarkReport(
        tokens_source=tokens_source,
        tokens_compressed=tokens_compressed,
        ratio=ratio,
        cost_baseline=round(baseline_cost, 6),
        cost_compressed=round(compressed_total, 6),
        cost_savings_pct=round(savings_pct, 2),
        quality_score=quality_score,
        quality_rationale=quality_rationale,
        layers_run=result.layers_run,
        extras={
            "extract_cost": round(extract_cost, 6),
            "baseline_out_tokens": baseline_out_tokens,
            "compressed_out_tokens": compressed_out_tokens,
            "stats": result.stats,
        },
    )
