"""Deterministic fake LLM used by unit and integration tests."""

from __future__ import annotations

from collections.abc import Iterator


class FakeLLM:
    """Minimal text-generation fake with queued responses and call history."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses or [])
        self.calls: list[dict[str, object]] = []

    def queue_response(self, response: str) -> None:
        """Append a response to the internal queue."""

        self._responses.append(response)

    def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:
        """Return a deterministic response for the given prompt."""

        self.calls.append(
            {
                "method": "generate",
                "prompt": prompt,
                "system_prompt": system_prompt,
            }
        )
        if self._responses:
            return self._responses.pop(0)
        return f"fake-llm-response:{prompt}"

    def stream(self, prompt: str, *, system_prompt: str | None = None) -> Iterator[str]:
        """Yield the generated response as whitespace-delimited chunks."""

        response = self.generate(prompt, system_prompt=system_prompt)
        for token in response.split():
            yield token
