from __future__ import annotations

from typing import Any

from ragms.libs.abstractions import BaseSplitter
from ragms.libs.providers.splitters.recursive_character_splitter import RecursiveCharacterSplitter


class SplitterFactory:
    _REGISTRY = {
        "recursive_character": RecursiveCharacterSplitter,
    }

    @classmethod
    def create(cls, config: dict[str, Any]) -> BaseSplitter:
        provider = config.get("provider")
        try:
            splitter_class = cls._REGISTRY[provider]
        except KeyError as exc:
            raise ValueError(f"Unknown splitter provider: {provider}") from exc
        return splitter_class(
            chunk_size=config.get("chunk_size", 900),
            chunk_overlap=config.get("chunk_overlap", 150),
            separators=config.get("separators"),
        )

