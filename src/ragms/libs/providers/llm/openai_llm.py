from __future__ import annotations

from typing import Any

from ragms.libs.abstractions import BaseLLM


class OpenAILLM(BaseLLM):
    def __init__(self, *, model: str, api_key: str | None = None, temperature: float | None = None) -> None:
        self.model = model
        self.api_key = api_key
        self.temperature = temperature

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        return {
            "provider": "openai",
            "model": self.model,
            "content": f"openai:{len(messages)}",
            "temperature": self.temperature,
            **kwargs,
        }

