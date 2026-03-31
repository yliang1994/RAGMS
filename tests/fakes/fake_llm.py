from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FakeLLM:
    response_text: str = "fake-llm-response"
    calls: list[dict[str, object]] = field(default_factory=list)

    def chat(self, messages: list[dict[str, str]], **kwargs: object) -> dict[str, object]:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return {
            "content": self.response_text,
            "model": "fake-llm",
            "usage": {"prompt_tokens": len(messages), "completion_tokens": 1},
        }

