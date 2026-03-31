from __future__ import annotations

from typing import Any

from ragms.libs.abstractions import BaseLoader


class MarkItDownLoader(BaseLoader):
    def __init__(self, *, extract_images: bool = True, output_format: str = "markdown") -> None:
        self.extract_images = extract_images
        self.output_format = output_format

    def load(self, source: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "source": source,
            "content": "",
            "metadata": {
                "loader": "markitdown",
                "extract_images": self.extract_images,
                "output_format": self.output_format,
                **kwargs,
            },
        }

