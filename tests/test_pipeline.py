"""Pipeline integration tests that do NOT call any external LLM.

We construct a Compressor with only the free layers (preprocess + codebook)
and verify the orchestrator returns a sensible CompressionResult.
"""

from __future__ import annotations

from narrato.pipeline import Compressor, Decoder


def _sample_norwegian_text() -> str:
    return (
        "Olsen-familien bodde på en gård i Hedmark. "
        "Olsen-familien hadde drevet gården i tre generasjoner. "
        "Olsen-familien var kjent i bygda for sin gjestfrihet. "
        "Per var den eldste sønnen og skulle overta gården. "
        "Per drømte om å bli ingeniør i Oslo. "
        "Han fortalte ingen om denne drømmen. "
        "Han fortalte ingen om denne drømmen i mange år. "
        "En kveld i mai bestemte Per seg for å snakke med faren."
    )


def test_compressor_runs_free_layers_only() -> None:
    compressor = Compressor(
        source_lang="no",
        provider="anthropic",  # only used for tokenizer here
        layers=["preprocess", "codebook"],
    )
    text = _sample_norwegian_text()
    result = compressor.compress(text)

    assert "preprocess" in result.layers_run
    assert "codebook" in result.layers_run
    assert result.format == "text"
    assert result.stats["input_tokens"] > 0
    assert result.stats["output_tokens"] > 0


def test_decoder_builds_prompt() -> None:
    compressor = Compressor(
        source_lang="no",
        provider="anthropic",
        layers=["preprocess", "codebook"],
    )
    result = compressor.compress(_sample_norwegian_text())
    prompt = Decoder.unpack_prompt(
        result,
        instruction="Skriv en kort fortelling.",
        target_lang="no",
    )
    assert "INSTRUCTION" in prompt
    assert "Skriv en kort fortelling." in prompt


def test_decoder_unpack_messages_splits_legend_into_system() -> None:
    compressor = Compressor(
        source_lang="no",
        provider="anthropic",
        layers=["preprocess", "codebook"],
    )
    result = compressor.compress(_sample_norwegian_text())
    msgs = Decoder.unpack_messages(result, instruction="Skriv en fortelling.")
    assert "system" in msgs and "user" in msgs
    if result.legend:
        assert "LEGEND" in msgs["system"]
    assert "Skriv en fortelling." in msgs["user"]


def test_unknown_layer_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        Compressor(layers=["preprocess", "bogus"])  # type: ignore[arg-type]
