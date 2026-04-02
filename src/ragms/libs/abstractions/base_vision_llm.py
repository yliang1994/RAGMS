"""Abstract vision-language model contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseVisionLLM(ABC):
    """Generate captions or descriptions from image inputs."""

    @abstractmethod
    def caption(
        self,
        image_path: str | Path,
        *,
        prompt: str | None = None,
        context: str | None = None,
    ) -> str:
        """Generate a caption for a single image."""

    @abstractmethod
    def caption_batch(
        self,
        image_paths: list[str | Path],
        *,
        prompt: str | None = None,
        context: str | None = None,
    ) -> list[str]:
        """Generate captions for multiple images."""

