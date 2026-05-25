# Type checking

`narratoflow` is fully annotated and ships a [PEP 561](https://peps.python.org/pep-0561/) `py.typed` marker. Type checkers pick up the inline annotations automatically — no separate stubs package, no `-py.typed` extra.

## Tested with

- **mypy** — clean import
- **pyright / pylance** — full inference, hover info, autocomplete
- **PyCharm built-in checker** — annotations recognised

## Quick check

```bash
pip install mypy
mypy your_app.py
```

`narrato`'s own checked types include:

- `Compressor`, `CompressionResult`, `Decoder` (all generic-free, plain dataclasses)
- `Provider`, `AsyncProvider`, `ProviderResponse` Protocols
- `Profile` dataclass and registry helpers
- `NarrativeFacts`, `QAFacts`, `InterviewFacts`, `DialogueFacts`, `NewsFacts` (Pydantic v2 BaseModels)

## Custom schema typing

When you pass a custom Pydantic model to `Compressor(schema=...)`, the runtime validates the extractor's output against your model. mypy will not yet narrow `result.payload` to your schema — `payload` stays a plain `dict[str, Any]` because the extractor returns whatever the LLM emitted. Use `MySchema.model_validate(result.payload)` to recover the typed object:

```python
from pydantic import BaseModel

class MyFacts(BaseModel):
    summary: str
    items: list[str]

c = Compressor(schema=MyFacts, ...)
result = c.compress(text)
facts: MyFacts = MyFacts.model_validate(result.payload)
```

## Async types

`Compressor.acompress` is `async def`; type checkers infer the return type as `Coroutine[..., CompressionResult]`. The provider parameter is typed as `Provider | AsyncProvider | str`. All ships providers satisfy both protocols.

## Strict mode

The codebase is written to be `mypy --strict`-friendly but does not currently CI-enforce strict mode. PRs improving strictness are welcome — see `pyproject.toml`'s `[tool.mypy]` section for the current configuration.
