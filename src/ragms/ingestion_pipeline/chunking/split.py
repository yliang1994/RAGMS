"""Chunking pipeline that normalizes splitter output into stable chunk records."""

from __future__ import annotations

import re
from typing import Any

from ragms.core.models import Chunk
from ragms.libs.abstractions import BaseSplitter


_IMAGE_REF_PATTERN = re.compile(r"\[IMAGE:\s*([^\]]+?)\s*\]")


class ChunkingPipeline:
    """Convert canonical documents into normalized chunk records."""

    def __init__(self, splitter: BaseSplitter) -> None:
        self.splitter = splitter

    def run(
        self,
        document: dict[str, Any],
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[Chunk]:
        """Split a canonical document and normalize chunks for downstream stages."""

        split_chunks = self.splitter.split(
            document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        document_metadata = dict(document.get("metadata") or {})
        document_id = str(document_metadata.get("document_id", ""))
        source_sha256 = str(document_metadata.get("source_sha256", ""))
        source_path = str(document.get("source_path") or document.get("source") or "")
        known_image_ids = {
            str(image.get("image_id"))
            for image in list(document_metadata.get("images") or [])
            if isinstance(image, dict) and image.get("image_id")
        }

        chunks: list[Chunk] = []
        for chunk in split_chunks:
            content = str(chunk.get("content", ""))
            referenced_images = [
                image_id
                for image_id in _IMAGE_REF_PATTERN.findall(content)
                if not known_image_ids or image_id in known_image_ids
            ]
            normalized_metadata = dict(document_metadata)
            normalized_metadata.update(dict(chunk.get("metadata") or {}))
            normalized_chunk = Chunk.from_splitter_chunk(
                {
                    **dict(chunk),
                    "metadata": normalized_metadata,
                },
                document_id=document_id,
                source_path=source_path,
                source_sha256=source_sha256,
                image_refs=referenced_images,
            )
            chunks.append(normalized_chunk)
        return chunks
