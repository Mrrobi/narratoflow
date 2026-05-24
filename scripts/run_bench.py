"""Loads .env then runs the narrato benchmark on the bundled Norwegian sample."""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from narrato.benchmark import run_benchmark
from narrato.pipeline import Compressor


def main(skip_quality: bool = False) -> int:
    sample = Path(__file__).parent.parent / "benchmarks" / "samples" / "norwegian_sample.txt"
    text = sample.read_text(encoding="utf-8")

    compressor = Compressor(
        source_lang="no",
        provider="openai",
        extractor_model="gpt-4o-mini",
        target_model="gpt-4o",
        schema="narrative",
        layers=["preprocess", "codebook", "extract"],
    )

    report = run_benchmark(
        text,
        instruction="Skriv en kort fortelling (200-300 ord) på norsk basert på fakta.",
        compressor=compressor,
        target_model="gpt-4o",
        judge_model="gpt-4o-mini",
        skip_quality=skip_quality,
    )

    print(report.to_json())
    return 0


if __name__ == "__main__":
    skip = "--skip-quality" in sys.argv
    raise SystemExit(main(skip_quality=skip))
