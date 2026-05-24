"""Layer 3 — semantic extraction via a small/cheap LLM.

The extractor takes the (already preprocessed + codebook'd) text and asks a
cheap model to fill a Pydantic schema. Output is a compact JSON payload — the
densest representation `narrato` can produce without learned compression.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from narrato.providers.base import Provider
from narrato.schemas import schema_to_json_schema

_SYSTEM = (
    "You extract facts from source text and emit them as JSON conforming to the "
    "provided schema. Preserve proper nouns and verbatim quotes in the SOURCE language. "
    "Be exhaustive but concise — do not invent details. If a field is unknown, leave it "
    "empty or null. Output only the JSON object, no commentary."
)


@dataclass
class ExtractResult:
    payload: dict[str, Any]
    raw_text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    valid: bool = True
    validation_error: str | None = None
    stats: dict = field(default_factory=dict)


def extract(
    text: str,
    schema: type[BaseModel],
    provider: Provider,
    model: str,
    *,
    legend: dict[str, str] | None = None,
    source_lang: str = "no",
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> ExtractResult:
    legend_block = ""
    if legend:
        lines = [f"  {code} = {phrase}" for code, phrase in legend.items()]
        legend_block = "The source text uses these substitution codes:\n" + "\n".join(lines) + "\n\n"

    user = (
        f"{legend_block}"
        f"Source language: {source_lang}\n\n"
        f"SOURCE TEXT:\n```\n{text}\n```\n\n"
        f"Extract the facts and emit them as a JSON object matching the schema."
    )

    json_schema = schema_to_json_schema(schema)
    resp = provider.complete_json(
        system=_SYSTEM,
        user=user,
        model=model,
        schema=json_schema,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    valid = True
    err: str | None = None
    try:
        payload = json.loads(resp.text)
    except Exception as e:
        payload = {"_raw": resp.text}
        valid = False
        err = f"json decode: {e}"

    if valid:
        try:
            schema.model_validate(payload)
        except ValidationError as e:
            valid = False
            err = str(e)

    return ExtractResult(
        payload=payload,
        raw_text=resp.text,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
        model=resp.model,
        valid=valid,
        validation_error=err,
        stats={"chars_out": len(resp.text)},
    )
