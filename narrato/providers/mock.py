"""Mock provider for offline tests and demos.

Use with ``Compressor(provider=MockProvider(payload=...))`` after constructing
the compressor — or pass ``MockProvider`` instances directly to ``extract`` and
``run_benchmark``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from narrato.providers.base import ProviderResponse


@dataclass
class MockProvider:
    """Returns canned responses; counts how many times each method ran.

    Parameters
    ----------
    payload:
        Default JSON payload returned by ``complete_json``. May be a dict or
        a callable taking the user prompt and returning a dict.
    text:
        Default text returned by ``complete``. May be a string or a callable.
    sequence:
        Optional list of canned ``ProviderResponse`` objects returned in order
        on successive calls (round-robin if exhausted). Overrides ``payload``
        and ``text`` when set.
    input_tokens / output_tokens:
        Token counts reported on each response.
    """

    name: str = "mock"
    payload: Any = field(default_factory=dict)
    text: Any = "ok"
    sequence: list[ProviderResponse] | None = None
    input_tokens: int = 100
    output_tokens: int = 50
    calls_complete: int = 0
    calls_complete_json: int = 0
    last_user: str = ""
    last_system: str = ""

    def _next_response(self) -> ProviderResponse | None:
        if not self.sequence:
            return None
        idx = (self.calls_complete + self.calls_complete_json - 1) % len(self.sequence)
        return self.sequence[idx]

    def complete(
        self,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> ProviderResponse:
        self.calls_complete += 1
        self.last_user = user
        self.last_system = system
        seq = self._next_response()
        if seq is not None:
            return seq
        text = self.text(user) if callable(self.text) else str(self.text)
        return ProviderResponse(
            text=text,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            model=model,
        )

    def complete_json(
        self,
        system: str,
        user: str,
        model: str,
        schema: dict[str, Any] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> ProviderResponse:
        self.calls_complete_json += 1
        self.last_user = user
        self.last_system = system
        seq = self._next_response()
        if seq is not None:
            return seq
        payload = self.payload(user) if callable(self.payload) else self.payload
        return ProviderResponse(
            text=json.dumps(payload, ensure_ascii=False),
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            model=model,
        )
