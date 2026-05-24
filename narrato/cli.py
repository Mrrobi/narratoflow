"""``narrato`` command-line interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from narrato.benchmark import run_benchmark
from narrato.pipeline import Compressor, Decoder


@click.group()
@click.version_option()
def main() -> None:
    """narrato — compress huge LLM input context."""


@main.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", "out_path", type=click.Path(path_type=Path), default=None, help="Output JSON path (default: stdout)")
@click.option("--lang", default="no", show_default=True)
@click.option("--provider", type=click.Choice(["anthropic", "openai"]), default="anthropic", show_default=True)
@click.option("--extractor-model", default="claude-haiku-4-5-20251001", show_default=True)
@click.option("--target-model", default="claude-opus-4-7", show_default=True)
@click.option("--schema", default="narrative", show_default=True)
@click.option(
    "--layer",
    "layers",
    multiple=True,
    type=click.Choice(["preprocess", "codebook", "extract"]),
    help="Repeat to enable multiple layers. Default: preprocess+codebook+extract.",
)
def compress(
    input_path: Path,
    out_path: Path | None,
    lang: str,
    provider: str,
    extractor_model: str,
    target_model: str,
    schema: str,
    layers: tuple[str, ...],
) -> None:
    """Compress INPUT_PATH and emit a JSON envelope."""
    text = input_path.read_text(encoding="utf-8")
    chosen_layers = list(layers) if layers else ["preprocess", "codebook", "extract"]
    compressor = Compressor(
        source_lang=lang,
        provider=provider,
        extractor_model=extractor_model,
        target_model=target_model,
        schema=schema,
        layers=chosen_layers,
    )
    result = compressor.compress(text)
    out_json = result.to_json()
    if out_path:
        out_path.write_text(out_json, encoding="utf-8")
        click.echo(f"wrote {out_path}  (tokens_in={result.stats.get('input_tokens')}, "
                   f"tokens_out={result.stats.get('output_tokens')}, "
                   f"ratio={result.stats.get('ratio'):.3f})")
    else:
        click.echo(out_json)


@main.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--target-task", required=True, help="Instruction passed to the downstream LLM, e.g. 'Skriv en kort fortelling.'")
@click.option("--lang", default="no", show_default=True)
@click.option("--provider", type=click.Choice(["anthropic", "openai"]), default="anthropic", show_default=True)
@click.option("--extractor-model", default="claude-haiku-4-5-20251001", show_default=True)
@click.option("--target-model", default="claude-opus-4-7", show_default=True)
@click.option("--judge-model", default=None, help="Defaults to extractor model.")
@click.option("--schema", default="narrative", show_default=True)
@click.option("--skip-quality", is_flag=True, help="Skip the LLM-judge step (no narrative generation).")
def eval(
    input_path: Path,
    target_task: str,
    lang: str,
    provider: str,
    extractor_model: str,
    target_model: str,
    judge_model: str | None,
    schema: str,
    skip_quality: bool,
) -> None:
    """Benchmark compression: tokens, cost savings, and (optionally) LLM-judge quality."""
    text = input_path.read_text(encoding="utf-8")
    compressor = Compressor(
        source_lang=lang,
        provider=provider,
        extractor_model=extractor_model,
        target_model=target_model,
        schema=schema,
    )
    report = run_benchmark(
        text,
        instruction=target_task,
        compressor=compressor,
        target_model=target_model,
        judge_model=judge_model,
        skip_quality=skip_quality,
    )
    click.echo(report.to_json())


@main.command()
@click.argument("compressed_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--instruction", required=True, help="Instruction to append after the compressed payload.")
@click.option("--target-lang", default=None)
def prompt(compressed_path: Path, instruction: str, target_lang: str | None) -> None:
    """Build a ready-to-send prompt from a `narrato compress` JSON file."""
    from narrato.pipeline import CompressionResult

    data = json.loads(compressed_path.read_text(encoding="utf-8"))
    result = CompressionResult(
        payload=data["payload"],
        legend=data.get("legend") or {},
        format=data.get("format", "text"),
        layers_run=data.get("layers_run") or [],
        stats=data.get("stats") or {},
    )
    click.echo(Decoder.unpack_prompt(result, instruction=instruction, target_lang=target_lang))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
