"""Define your own Pydantic schema and pass it to the Compressor."""

from __future__ import annotations

from pydantic import BaseModel, Field

from narrato import Compressor


class InterviewFacts(BaseModel):
    interviewer: str | None = None
    interviewee: str | None = None
    main_topic: str
    key_points: list[str] = Field(default_factory=list)
    quotes: list[str] = Field(default_factory=list)
    sentiment: str | None = None


if __name__ == "__main__":
    text = (
        "Reporter Anne Lien snakket med statsråd Per Olsen om budsjettet for 2026. "
        "Olsen forklarte at hovedfokuset er på utdanning og helsevesen. "
        "'Vi må prioritere de yngste og de eldste,' sa Olsen. "
        "Anne spurte om skatteendringer; Olsen svarte at det ikke vil komme nye skatter i 2026."
    )
    c = Compressor(
        source_lang="no",
        provider="anthropic",
        extractor_model="claude-haiku-4-5-20251001",
        schema=InterviewFacts,
        layers=["preprocess", "extract"],
    )
    res = c.compress(text)
    print(res.to_json())
