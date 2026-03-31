from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FakeVisionLLM:
    default_caption: str = "fake-image-caption"
    calls: list[dict[str, object]] = field(default_factory=list)

    def caption(self, image_ref: str, prompt: str | None = None) -> dict[str, str]:
        self.calls.append({"image_ref": image_ref, "prompt": prompt})
        return {
            "caption": self.default_caption,
            "image_ref": image_ref,
        }

