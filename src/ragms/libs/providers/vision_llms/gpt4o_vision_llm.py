"""Lightweight GPT-4o vision provider placeholder."""

from __future__ import annotations

from pathlib import Path

from ragms.libs.abstractions import BaseVisionLLM


class GPT4oVisionLLM(BaseVisionLLM):
    """Generate deterministic captions for GPT-4o-style image understanding."""

    provider_name = "gpt4o"

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

    def caption(
        self,
        image_path: str | Path,
        *,
        prompt: str | None = None,
        context: str | None = None,
    ) -> str:
        """Generate a deterministic caption for a single image."""

        path = Path(image_path)
        prompt_suffix = f":{prompt}" if prompt else ""
        context_suffix = f":{context}" if context else ""
        return f"{self.provider_name}:{self.model}:{path.stem}{prompt_suffix}{context_suffix}"

    def caption_batch(
        self,
        image_paths: list[str | Path],
        *,
        prompt: str | None = None,
        context: str | None = None,
    ) -> list[str]:
        """Generate deterministic captions for multiple images."""

        return [
            self.caption(image_path, prompt=prompt, context=context)
            for image_path in image_paths
        ]
