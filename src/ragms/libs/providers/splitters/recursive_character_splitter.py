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
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must not be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def split(self, document: dict[str, Any], **kwargs: Any) -> list[dict[str, Any]]:
        text = str(document.get("content", ""))
        base_metadata = dict(document.get("metadata", {}))
        if not text:
            return []

        step = self.chunk_size - self.chunk_overlap
        chunks: list[dict[str, Any]] = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            chunks.append(
                {
                    "chunk_id": f"{kwargs.get('chunk_prefix', 'chunk')}-{chunk_index}",
                    "text": chunk_text,
                    "metadata": {
                        **base_metadata,
                        "splitter": "recursive_character",
                        "chunk_index": chunk_index,
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": self.chunk_overlap,
                        "separators": self.separators,
                        "start_offset": start,
                        "end_offset": end,
                    },
                }
            )
            if end >= len(text):
                break
            start += step
            chunk_index += 1

        return chunks

