from __future__ import annotations

from ragms.libs.abstractions import BaseVisionLLM


class GPT4OVisionLLM(BaseVisionLLM):
    def __init__(self, *, model: str, api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key

    def caption(self, image_ref: str, prompt: str | None = None) -> dict[str, str]:
        return {
            "provider": "openai",
            "model": self.model,
            "image_ref": image_ref,
            "caption": prompt or "gpt4o-vision-caption",
        }

