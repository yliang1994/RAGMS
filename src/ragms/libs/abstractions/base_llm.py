from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseLLM(ABC):
    @abstractmethod
    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        """Run a chat completion request and return a normalized response."""

