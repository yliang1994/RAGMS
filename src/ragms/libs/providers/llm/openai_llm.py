from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from ragms.libs.abstractions import BaseLLM


class OpenAILLM(BaseLLM):
    SUPPORTED_MODELS = {
        "gpt-4.1-mini",
        "gpt-4.1",
        "gpt-4o",
        "gpt-5",
    }

    def __init__(self, *, model: str, api_key: str | None = None, temperature: float | None = None) -> None:
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self._validate_configuration()

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        return self.generate(messages, **kwargs)

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        self._ensure_messages(messages)
        content = self._build_response_text(messages)
        return {
            "provider": "openai",
            "model": self.model,
            "content": content,
            "temperature": self.temperature,
            "usage": {
                "prompt_tokens": len(messages),
                "completion_tokens": max(1, len(content.split())),
            },
            **kwargs,
        }

    def stream(self, messages: list[dict[str, str]], **kwargs: Any) -> Iterator[dict[str, Any]]:
        response = self.generate(messages, **kwargs)
        for token in response["content"].split():
            yield {
                "provider": "openai",
                "model": self.model,
                "delta": token,
            }

    def _validate_configuration(self) -> None:
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        if self.model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported OpenAI model: {self.model}")

    def _ensure_messages(self, messages: list[dict[str, str]]) -> None:
        if not messages:
            raise ValueError("messages must not be empty")
        if any("content" not in message for message in messages):
            raise RuntimeError("Upstream response error: malformed message payload")

    def _build_response_text(self, messages: list[dict[str, str]]) -> str:
        last_message = messages[-1]["content"].strip()
        return f"openai-response: {last_message or '<empty>'}"
