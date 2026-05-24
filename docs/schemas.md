# Schemas

Schemas tell the extractor *what to keep* from the source text. They are plain Pydantic models — bring your own, or use a preset.

## Built-in presets

| preset | when |
|---|---|
| `narrative` | story/narrative generation. Characters, setting, ordered events, themes, tone, verbatim quotes. |
| `qa` | fact-extraction / RAG. Summary, entities, dates, claims. |
| `interview` | interview / transcript. Interviewer + interviewee, ordered turns with speaker/summary/quote, key points, sentiment. |
| `dialogue` | scripted / fictional dialogue. Participants, setting, ordered lines with speaker/line/intent, arc, notable quotes. |
| `news` | news article. Headline, lede, 5W1H (who/what/when/where/why/how), sources, quotes. |

```python
from narrato import Compressor

c = Compressor(schema="narrative", ...)
```

## `NarrativeFacts`

```python
class Character(BaseModel):
    name: str
    role: str | None = None
    traits: list[str] = []

class Setting(BaseModel):
    place: str | None = None
    time: str | None = None
    atmosphere: str | None = None

class Event(BaseModel):
    when: str | None = None
    who: list[str] = []
    action: str
    outcome: str | None = None

class NarrativeFacts(BaseModel):
    characters: list[Character] = []
    setting: Setting = Setting()
    events: list[Event] = []
    themes: list[str] = []
    tone: str | None = None
    key_quotes: list[str] = []      # verbatim, source language
```

## Custom schemas

Pass any Pydantic v2 `BaseModel`:

```python
from pydantic import BaseModel, Field
from narrato import Compressor

class InterviewFacts(BaseModel):
    interviewer: str | None = None
    interviewee: str | None = None
    main_topic: str
    key_points: list[str] = Field(default_factory=list)
    quotes: list[str] = Field(default_factory=list)
    sentiment: str | None = None

c = Compressor(schema=InterviewFacts, ...)
```

## Design tips

- **Keep verbatim fields**. If you need quotes or proper-noun spellings preserved, give them a list field. The extractor is told to copy them verbatim in the source language.
- **Be exhaustive but flat**. Deep nested objects cost more output tokens than wide flat ones.
- **Use `Optional` / defaults** for fields that may be missing — the extractor is instructed not to invent.
- **Order matters for events**. The schema documentation `description` on a list field nudges the extractor to preserve order.

## Inspecting the generated JSON schema

```python
from narrato.schemas import schema_to_json_schema, NarrativeFacts

print(schema_to_json_schema(NarrativeFacts))
```

This is what gets passed to OpenAI's `response_format` (json_schema) and Anthropic's tool-use input_schema.
