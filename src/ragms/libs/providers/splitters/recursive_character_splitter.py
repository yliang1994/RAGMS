from __future__ import annotations

from typing import Any

from ragms.libs.abstractions import BaseSplitter


class RecursiveCharacterSplitter(BaseSplitter):
    def __init__(
        self,
        *,
        chunk_size: int = 900,
        chunk_overlap: int = 150,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def split(self, document: dict[str, Any], **kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                "chunk_id": kwargs.get("chunk_id", "chunk-0"),
                "text": document.get("content", ""),
                "metadata": {
                    "splitter": "recursive_character",
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "separators": self.separators,
                },
            }
        ]

