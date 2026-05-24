"""Pipeline integration tests.

Most tests run only the free layers (preprocess + codebook) so they need no
LLM. Where the extract layer is exercised, we inject a MockProvider.
"""

from __future__ import annotations

from narrato.pipeline import Compressor, Decoder
from narrato.providers.mock import MockProvider


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
    result = compressor.compress(_sample_norwegian_text())
    assert "preprocess" in result.layers_run
    assert "codebook" in result.layers_run
    assert result.format == "text"
    assert result.stats["input_tokens"] > 0
    assert result.stats["output_tokens"] > 0
    assert result.stats["cache_enabled"] is False


def test_decoder_builds_prompt() -> None:
    compressor = Compressor(provider="anthropic", layers=["preprocess", "codebook"])
    result = compressor.compress(_sample_norwegian_text())
    prompt = Decoder.unpack_prompt(
        result, instruction="Skriv en kort fortelling.", target_lang="no"
    )
    assert "INSTRUCTION" in prompt
    assert "Skriv en kort fortelling." in prompt


def test_decoder_unpack_messages_splits_legend_into_system() -> None:
    compressor = Compressor(provider="anthropic", layers=["preprocess", "codebook"])
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


def test_extract_layer_with_mock_provider() -> None:
    canned = {
        "characters": [{"name": "Per"}],
        "events": [{"action": "talk"}],
        "themes": ["family"],
        "key_quotes": [],
    }
    mock = MockProvider(payload=canned)
    compressor = Compressor(
        provider=mock,
        layers=["preprocess", "codebook", "extract"],
        schema="narrative",
        chunked=False,
    )
    result = compressor.compress(_sample_norwegian_text())
    assert "extract" in result.layers_run
    assert result.format == "json"
    assert result.stats["extract"]["mode"] == "single"
    assert mock.calls_complete_json == 1


def test_chunked_extract_triggers_for_long_text() -> None:
    long_text = ". ".join(f"Sentence {i}" for i in range(400)) + "."
    mock = MockProvider(payload={"what": "fact", "who": ["x"]})
    compressor = Compressor(
        provider=mock,
        layers=["extract"],
        schema="news",
        chunk_chars=500,
        overlap_chars=20,
    )
    result = compressor.compress(long_text)
    assert result.stats["extract"]["mode"] == "chunked"
    assert result.stats["extract"]["chunks"] > 1
    assert mock.calls_complete_json == result.stats["extract"]["chunks"]


def test_cache_flag_propagates_to_provider() -> None:
    """Smoke test: cache=True doesn't break the offline free-layer path."""
    compressor = Compressor(
        provider="anthropic",
        layers=["preprocess", "codebook"],
        cache=True,
    )
    result = compressor.compress(_sample_norwegian_text())
    assert result.stats["cache_enabled"] is True
