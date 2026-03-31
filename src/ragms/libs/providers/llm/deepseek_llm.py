from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from ragms.libs.abstractions import BaseLLM


class DeepSeekLLM(BaseLLM):
    SUPPORTED_MODELS = {
        "deepseek-chat",
        "deepseek-reasoner",
    }

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        temperature: float | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.base_url = base_url
        self._validate_configuration()

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        return self.generate(messages, **kwargs)

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        self._ensure_messages(messages)
        if kwargs.get("simulate_rate_limit"):
            raise RuntimeError("DeepSeek rate limit exceeded")
        if kwargs.get("simulate_upstream_failure"):
            raise RuntimeError("DeepSeek upstream request failed")
        content = self._build_response_text(messages)
        return {
            "provider": "deepseek",
            "model": self.model,
            "content": content,
            "temperature": self.temperature,
            "base_url": self.base_url,
            **kwargs,
        }

    def stream(self, messages: list[dict[str, str]], **kwargs: Any) -> Iterator[dict[str, Any]]:
        response = self.generate(messages, **kwargs)
        for token in response["content"].split():
            yield {
                "provider": "deepseek",
                "model": self.model,
                "delta": token,
                "base_url": self.base_url,
            }

    def _validate_configuration(self) -> None:
        if not self.api_key:
            raise ValueError("DeepSeek API key is required")
        if self.model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported DeepSeek model: {self.model}")

    def _ensure_messages(self, messages: list[dict[str, str]]) -> None:
        if not messages:
            raise ValueError("messages must not be empty")
        if any("content" not in message for message in messages):
            raise RuntimeError("DeepSeek upstream request failed: malformed message payload")

    def _build_response_text(self, messages: list[dict[str, str]]) -> str:
        return f"deepseek-response: {messages[-1]['content'].strip() or '<empty>'}"
