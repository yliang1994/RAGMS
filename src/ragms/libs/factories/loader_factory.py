from __future__ import annotations

from typing import Any

from ragms.libs.abstractions import BaseLoader
from ragms.libs.providers.loaders.markitdown_loader import MarkItDownLoader


class LoaderFactory:
    _REGISTRY = {
        "markitdown": MarkItDownLoader,
    }

    @classmethod
    def create(cls, config: dict[str, Any]) -> BaseLoader:
        provider = config.get("provider")
        try:
            loader_class = cls._REGISTRY[provider]
        except KeyError as exc:
            raise ValueError(f"Unknown loader provider: {provider}") from exc
        return loader_class(
            extract_images=config.get("extract_images", True),
            output_format=config.get("output_format", "markdown"),
        )

