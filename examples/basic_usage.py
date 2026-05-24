"""End-to-end example: compress a Norwegian story, generate a narrative.

Requires:
    pip install narrato
    export ANTHROPIC_API_KEY=...

Run:
    python examples/basic_usage.py
"""

from __future__ import annotations

import os
from pathlib import Path

from narrato import Compressor, Decoder

SAMPLE = Path(__file__).parent.parent / "benchmarks" / "samples" / "norwegian_sample.txt"


def main() -> None:
    text = SAMPLE.read_text(encoding="utf-8")

    compressor = Compressor(
        source_lang="no",
        provider="anthropic",
        extractor_model="claude-haiku-4-5-20251001",
        target_model="claude-opus-4-7",
        layers=["preprocess", "codebook", "extract"],
        schema="narrative",
    )

    result = compressor.compress(text)
    print("=== compression stats ===")
    for k, v in result.stats.items():
        print(f"{k}: {v}")
    print()

    prompt = Decoder.unpack_prompt(
        result,
        instruction="Skriv en kort fortelling (ca 200 ord) på norsk basert på fakta.",
        target_lang="no",
    )
    print("=== final prompt (truncated) ===")
    print(prompt[:1200])
    print("..." if len(prompt) > 1200 else "")

    if os.getenv("ANTHROPIC_API_KEY"):
        from anthropic import Anthropic

        client = Anthropic()
        resp = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=600,
            system="Du er en dyktig forfatter på norsk.",
            messages=[{"role": "user", "content": prompt}],
        )
        narrative = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        print("\n=== generated narrative ===")
        print(narrative)


if __name__ == "__main__":
    main()
