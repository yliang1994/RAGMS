"""Abstract text-generation model contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator


class BaseLLM(ABC):
    """Generate text responses from prompt input."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> str:
        """Generate a complete text response."""

    @abstractmethod
    def stream(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> Iterator[str]:
        """Stream a text response token by token or chunk by chunk."""

