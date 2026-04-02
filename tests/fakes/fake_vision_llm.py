"""Deterministic fake vision-language model for image caption tests."""

from __future__ import annotations

from pathlib import Path


class FakeVisionLLM:
    """Return stable captions from image names and optional context."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def caption(
        self,
        image_path: str | Path,
        *,
        prompt: str | None = None,
        context: str | None = None,
    ) -> str:
        """Generate a deterministic caption for a single image."""

        path = Path(image_path)
        self.calls.append(
            {
                "method": "caption",
                "image_path": path,
                "prompt": prompt,
                "context": context,
            }
        )
        context_suffix = f" context={context}" if context else ""
        return f"caption:{path.stem}{context_suffix}"

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
