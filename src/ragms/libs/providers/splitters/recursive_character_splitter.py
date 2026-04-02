"""Deterministic character-based splitter used as the default placeholder provider."""

from __future__ import annotations

from typing import Any

from ragms.libs.abstractions import BaseSplitter


class RecursiveCharacterSplitter(BaseSplitter):
    """Split canonical documents into stable character windows."""

    def __init__(self, *, chunk_size: int = 1000, chunk_overlap: int = 100) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be between 0 and chunk_size - 1")
        self.default_chunk_size = chunk_size
        self.default_chunk_overlap = chunk_overlap

    def split(
        self,
        document: dict[str, Any],
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[dict[str, Any]]:
        """Split a canonical document into deterministic chunk records."""

        text = str(document.get("content", ""))
        if not text:
            return []

        resolved_chunk_size = chunk_size or self.default_chunk_size
        resolved_overlap = (
            self.default_chunk_overlap if chunk_overlap is None else chunk_overlap
        )
        if resolved_chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if resolved_overlap < 0 or resolved_overlap >= resolved_chunk_size:
            raise ValueError("chunk_overlap must be between 0 and chunk_size - 1")

        metadata = dict(document.get("metadata") or {})
        source = document.get("source") or document.get("source_path") or metadata.get("source")
        step = resolved_chunk_size - resolved_overlap
        chunks: list[dict[str, Any]] = []

        for chunk_index, start in enumerate(range(0, len(text), step)):
            end = min(start + resolved_chunk_size, len(text))
            chunks.append(
                {
                    "content": text[start:end],
                    "chunk_index": chunk_index,
                    "start_offset": start,
                    "end_offset": end,
                    "source": source,
                    "metadata": dict(metadata),
                }
            )
            if end >= len(text):
                break

        return chunks
