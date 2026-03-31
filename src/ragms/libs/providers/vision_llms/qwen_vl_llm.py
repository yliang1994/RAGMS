from __future__ import annotations

from typing import Any

from ragms.libs.abstractions import BaseVisionLLM


class QwenVLLM(BaseVisionLLM):
    SUPPORTED_MODELS = {
        "qwen-vl-max",
        "qwen-vl-plus",
    }

    def __init__(self, *, model: str, api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key
        self._validate_configuration()

    def caption(self, image_ref: str, prompt: str | None = None, **kwargs: Any) -> dict[str, str]:
        self._validate_image_ref(image_ref)
        if kwargs.get("simulate_model_unavailable"):
            raise RuntimeError("Qwen-VL model unavailable")
        low_quality = bool(kwargs.get("simulate_low_quality"))
        context = kwargs.get("context")
        return {
            "provider": "qwen",
            "model": self.model,
            "image_ref": image_ref,
            "caption": self._build_caption(prompt=prompt, context=context, low_quality=low_quality),
            "quality": "low" if low_quality else "ok",
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
            raise ValueError("Qwen-VL API key is required")
        if self.model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported Qwen-VL model: {self.model}")

    def _validate_image_ref(self, image_ref: str) -> None:
        if not image_ref or not image_ref.strip():
            raise ValueError("image_ref must not be empty")

    def _build_caption(self, *, prompt: str | None, context: str | None, low_quality: bool) -> str:
        if low_quality:
            return "qwen-vl-caption-insufficient-detail"
        parts = ["qwen-vl-caption"]
        if prompt:
            parts.append(prompt)
        if context:
            parts.append(f"context={context}")
        return " | ".join(parts)
