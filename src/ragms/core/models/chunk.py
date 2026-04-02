"""Structured chunk model with stable identifier generation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Chunk:
    """Normalized chunk record used by downstream ingestion stages."""

    chunk_id: str
    document_id: str
    content: str
    source_path: str
    chunk_index: int
    start_offset: int
    end_offset: int
    source_ref: str | None = None
    image_refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_splitter_chunk(
        cls,
        chunk: dict[str, Any],
        *,
        document_id: str,
        source_path: str,
        source_sha256: str,
        source_ref: str | None,
        image_refs: list[str],
        image_occurrences: list[dict[str, Any]] | None = None,
        images: list[dict[str, Any]] | None = None,
    ) -> "Chunk":
        """Build a normalized chunk from a splitter payload."""

        chunk_index = int(chunk.get("chunk_index", 0))
        start_offset = int(chunk.get("start_offset", 0))
        end_offset = int(chunk.get("end_offset", start_offset + len(str(chunk.get("content", "")))))
        content = str(chunk.get("content", ""))
        metadata = dict(chunk.get("metadata") or {})
        metadata["image_refs"] = list(image_refs)
        if image_occurrences:
            metadata["image_occurrences"] = [dict(item) for item in image_occurrences]
        else:
            metadata.pop("image_occurrences", None)
        if images:
            metadata["images"] = [dict(item) for item in images]
        else:
            metadata.pop("images", None)
        chunk_id = cls.build_id(
            source_sha256=source_sha256,
            chunk_index=chunk_index,
            start_offset=start_offset,
            end_offset=end_offset,
            content=content,
        )
        return cls(
            chunk_id=chunk_id,
            document_id=document_id,
            content=content,
            source_path=source_path,
            chunk_index=chunk_index,
            start_offset=start_offset,
            end_offset=end_offset,
            source_ref=source_ref,
            image_refs=list(image_refs),
            metadata=metadata,
        )

    @staticmethod
    def build_id(
        *,
        source_sha256: str,
        chunk_index: int,
        start_offset: int,
        end_offset: int,
        content: str,
    ) -> str:
        """Build a stable chunk identifier from source identity and chunk boundaries."""

        digest = hashlib.sha256(
            f"{source_sha256}:{chunk_index}:{start_offset}:{end_offset}:{content}".encode("utf-8")
        ).hexdigest()
        return f"chunk_{digest[:16]}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the chunk into the repository's dict-based interchange format."""

        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content": self.content,
            "source_path": self.source_path,
            "chunk_index": self.chunk_index,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "source_ref": self.source_ref,
            "image_refs": list(self.image_refs),
            "metadata": dict(self.metadata),
        }
