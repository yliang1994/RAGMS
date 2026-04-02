"""Lightweight OpenAI-compatible LLM provider placeholder."""

from __future__ import annotations

from collections.abc import Iterator

from ragms.libs.abstractions import BaseLLM


class OpenAILLM(BaseLLM):
    """Return deterministic text responses for the configured OpenAI model."""

    provider_name = "openai"

    def __init__(
        self,
        *,
        model: str = "gpt-4.1-mini",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:
        """Generate a deterministic response payload."""

        prefix = f"{self.provider_name}:{self.model}"
        if system_prompt:
            return f"{prefix}:{system_prompt}:{prompt}"
        return f"{prefix}:{prompt}"

    def stream(self, prompt: str, *, system_prompt: str | None = None) -> Iterator[str]:
        """Yield the generated response as whitespace-delimited chunks."""

        for token in self.generate(prompt, system_prompt=system_prompt).split():
            yield token
