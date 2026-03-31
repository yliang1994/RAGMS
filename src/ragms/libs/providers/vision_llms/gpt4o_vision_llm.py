from __future__ import annotations

from typing import Any

from ragms.libs.abstractions import BaseVisionLLM


class GPT4OVisionLLM(BaseVisionLLM):
    SUPPORTED_MODELS = {
        "gpt-4.1-mini",
        "gpt-4o",
    }

    def __init__(self, *, model: str, api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key
        self._validate_configuration()

    def caption(self, image_ref: str, prompt: str | None = None, **kwargs: Any) -> dict[str, str]:
        self._validate_image_ref(image_ref)
        if kwargs.get("simulate_upstream_failure"):
            raise RuntimeError("GPT-4o Vision upstream response error")
        context = kwargs.get("context")
        caption_text = self._build_caption(prompt=prompt, context=context)
        return {
            "provider": "openai",
            "model": self.model,
            "image_ref": image_ref,
            "caption": caption_text,
        }

    def caption_batch(
        self,
        image_refs: list[str],
        prompt: str | None = None,
        *,
        context: str | None = None,
    ) -> list[dict[str, str]]:
        return [self.caption(image_ref, prompt=prompt, context=context) for image_ref in image_refs]

    def _validate_configuration(self) -> None:
        if not self.api_key:
            raise ValueError("GPT-4o Vision API key is required")
        if self.model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported GPT-4o Vision model: {self.model}")

    def _validate_image_ref(self, image_ref: str) -> None:
        if not image_ref or not image_ref.strip():
            raise ValueError("image_ref must not be empty")
        if image_ref.startswith("data:") and ";base64," not in image_ref:
            raise ValueError("invalid image encoding")

    def _build_caption(self, *, prompt: str | None, context: str | None) -> str:
        parts = ["gpt4o-vision-caption"]
        if prompt:
            parts.append(prompt)
        if context:
            parts.append(f"context={context}")
        return " | ".join(parts)
