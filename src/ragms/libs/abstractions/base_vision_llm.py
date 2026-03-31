from __future__ import annotations

from abc import ABC, abstractmethod


class BaseVisionLLM(ABC):
    @abstractmethod
    def caption(self, image_ref: str, prompt: str | None = None) -> dict[str, str]:
        """Generate a caption or description for an image reference."""

