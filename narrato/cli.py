"""``narrato`` command-line interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from narrato.benchmark import run_benchmark
from narrato.pipeline import Compressor, Decoder
from narrato.profiles import list_profiles


@click.group()
@click.version_option()
def main() -> None:
    """narrato — compress huge LLM input context."""


@main.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", "out_path", type=click.Path(path_type=Path), default=None, help="Output JSON path (default: stdout)")
@click.option("--profile", default=None, help="Named profile to start from (see `narratoflow profiles`).")
@click.option("--lang", default=None, help="Source language ISO code; overrides profile.")
@click.option("--provider", type=click.Choice(["anthropic", "openai"]), default=None, help="Provider; overrides profile.")
@click.option("--extractor-model", default=None, help="Extractor model; overrides profile.")
@click.option("--target-model", default=None, help="Target model; overrides profile.")
@click.option("--schema", default=None, help="Schema preset name or 'module:Class' path.")
@click.option(
    "--layer",
    "layers",
    multiple=True,
    type=click.Choice(["preprocess", "codebook", "extract"]),
    help="Repeat to enable multiple layers. Default: preprocess+codebook+extract.",
)
@click.option("--cache/--no-cache", default=None, help="Enable provider-side prompt caching (Anthropic).")
@click.option("--chunked/--no-chunked", default=None, help="Force chunked extraction mode.")
@click.option("--chunk-chars", type=int, default=None, help="Chunk size in characters.")
def compress(
    input_path: Path,
    out_path: Path | None,
    profile: str | None,
    lang: str | None,
    provider: str | None,
    extractor_model: str | None,
    target_model: str | None,
    schema: str | None,
    layers: tuple[str, ...],
    cache: bool | None,
    chunked: bool | None,
    chunk_chars: int | None,
) -> None:
    """Compress INPUT_PATH and emit a JSON envelope."""
    text = input_path.read_text(encoding="utf-8")

    if profile:
        kwargs: dict = {}
        if provider is not None:
            kwargs["provider"] = provider
        if extractor_model is not None:
            kwargs["extractor_model"] = extractor_model
        if target_model is not None:
            kwargs["target_model"] = target_model
        if lang is not None:
            kwargs["source_lang"] = lang
        if schema is not None:
            kwargs["schema"] = schema
        if layers:
            kwargs["layers"] = list(layers)
        if cache is not None:
            kwargs["cache"] = cache
        if chunked is not None:
            kwargs["chunked"] = chunked
        if chunk_chars is not None:
            kwargs["chunk_chars"] = chunk_chars
        compressor = Compressor.from_profile(profile, **kwargs)
    else:
        compressor = Compressor(
            source_lang=lang or "en",
            provider=provider or "anthropic",
            extractor_model=extractor_model or "claude-haiku-4-5-20251001",
            target_model=target_model or "claude-opus-4-7",
            schema=schema or "qa",
            layers=list(layers) if layers else ["preprocess", "codebook", "extract"],
            cache=bool(cache),
            chunked=bool(chunked),
            chunk_chars=chunk_chars or 8000,
        )

    result = compressor.compress(text)
    out_json = result.to_json()
    if out_path:
        out_path.write_text(out_json, encoding="utf-8")
        click.echo(
            f"wrote {out_path}  (tokens_in={result.stats.get('input_tokens')}, "
            f"tokens_out={result.stats.get('output_tokens')}, "
            f"ratio={result.stats.get('ratio'):.3f})"
        )
    else:
        click.echo(out_json)


@main.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--target-task", required=True, help="Instruction passed to the downstream LLM.")
@click.option("--profile", default=None, help="Named profile to start from.")
@click.option("--lang", default=None)
@click.option("--provider", type=click.Choice(["anthropic", "openai"]), default=None)
@click.option("--extractor-model", default=None)
@click.option("--target-model", default=None)
@click.option("--judge-model", default=None, help="Defaults to extractor model.")
@click.option("--schema", default=None)
@click.option("--skip-quality", is_flag=True, help="Skip the LLM-judge step.")
def eval(
    input_path: Path,
    target_task: str,
    profile: str | None,
    lang: str | None,
    provider: str | None,
    extractor_model: str | None,
    target_model: str | None,
    judge_model: str | None,
    schema: str | None,
    skip_quality: bool,
) -> None:
    """Benchmark compression: tokens, cost savings, and (optionally) LLM-judge quality."""
    text = input_path.read_text(encoding="utf-8")

    if profile:
        kwargs: dict = {}
        if provider is not None:
            kwargs["provider"] = provider
        if extractor_model is not None:
            kwargs["extractor_model"] = extractor_model
        if target_model is not None:
            kwargs["target_model"] = target_model
        if lang is not None:
            kwargs["source_lang"] = lang
        if schema is not None:
            kwargs["schema"] = schema
        compressor = Compressor.from_profile(profile, **kwargs)
    else:
        compressor = Compressor(
            source_lang=lang or "en",
            provider=provider or "anthropic",
            extractor_model=extractor_model or "claude-haiku-4-5-20251001",
            target_model=target_model or "claude-opus-4-7",
            schema=schema or "qa",
        )

    report = run_benchmark(
        text,
        instruction=target_task,
        compressor=compressor,
        target_model=compressor.target_model,
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


@main.command()
def profiles() -> None:
    """List all available named profiles."""
    rows = list_profiles()
    width = max(len(p.name) for p in rows) + 2
    click.echo(f"{'NAME':<{width}}DESCRIPTION")
    click.echo(f"{'-' * (width - 1):<{width}}{'-' * 60}")
    for p in rows:
        click.echo(f"{p.name:<{width}}{p.description}")


@main.command(name="schemas")
def schemas_cmd() -> None:
    """List built-in schema presets."""
    from narrato.schemas import list_presets

    for name in list_presets():
        click.echo(name)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
