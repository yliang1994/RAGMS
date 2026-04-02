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
        document_images = self._index_document_images(document_metadata)
        document_occurrences = self._normalize_occurrences(document_metadata)

        chunks: list[Chunk] = []
        for fallback_chunk_index, chunk in enumerate(split_chunks):
            content = str(chunk.get("content", ""))
            start_offset = int(chunk.get("start_offset", 0))
            end_offset = int(chunk.get("end_offset", start_offset + len(content)))
            chunk_occurrences = [
                occurrence
                for occurrence in document_occurrences
                if self._occurrence_overlaps_chunk(occurrence, start_offset=start_offset, end_offset=end_offset)
            ]
            occurrence_refs = [str(occurrence["image_id"]) for occurrence in chunk_occurrences]
            placeholder_refs = [image_id.strip() for image_id in _IMAGE_REF_PATTERN.findall(content)]
            referenced_images = self._merge_image_refs(occurrence_refs, placeholder_refs)
            chunk_images = [
                dict(document_images[image_id])
                for image_id in referenced_images
                if image_id in document_images
            ]
            normalized_metadata = self._build_chunk_metadata(
                document_metadata=document_metadata,
                chunk_metadata=dict(chunk.get("metadata") or {}),
                chunk_index=int(chunk.get("chunk_index", fallback_chunk_index)),
                source_ref=document_id or source_path or None,
                image_refs=referenced_images,
                image_occurrences=chunk_occurrences,
                images=chunk_images,
            )
            normalized_chunk = Chunk.from_splitter_chunk(
                {
                    **dict(chunk),
                    "chunk_index": int(chunk.get("chunk_index", fallback_chunk_index)),
                    "metadata": normalized_metadata,
                },
                document_id=document_id,
                source_path=source_path,
                source_sha256=source_sha256,
                source_ref=document_id or source_path or None,
                image_refs=referenced_images,
                image_occurrences=chunk_occurrences,
                images=chunk_images,
            )
            chunks.append(normalized_chunk)
        return chunks

    @staticmethod
    def _index_document_images(document_metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Index document-level image metadata by image id."""

        indexed: dict[str, dict[str, Any]] = {}
        for raw_image in list(document_metadata.get("images") or []):
            if not isinstance(raw_image, dict):
                continue
            image_id = str(raw_image.get("id") or raw_image.get("image_id") or "").strip()
            if not image_id:
                continue
            indexed[image_id] = {
                "id": image_id,
                "path": str(raw_image.get("path") or ""),
                "page": raw_image.get("page"),
                "position": dict(raw_image.get("position") or {}),
            }
        return indexed

    @staticmethod
    def _normalize_occurrences(document_metadata: dict[str, Any]) -> list[dict[str, Any]]:
        """Normalize document-level image occurrences into a stable list."""

        normalized: list[dict[str, Any]] = []
        for raw_occurrence in list(document_metadata.get("image_occurrences") or []):
            if not isinstance(raw_occurrence, dict):
                continue
            image_id = str(raw_occurrence.get("image_id") or raw_occurrence.get("id") or "").strip()
            if not image_id:
                continue
            text_offset = int(raw_occurrence.get("text_offset", 0))
            text_length = int(raw_occurrence.get("text_length", 0))
            normalized.append(
                {
                    "image_id": image_id,
                    "text_offset": text_offset,
                    "text_length": text_length,
                    "page": raw_occurrence.get("page"),
                    "position": dict(raw_occurrence.get("position") or {}),
                }
            )
        normalized.sort(key=lambda item: (int(item["text_offset"]), str(item["image_id"])))
        return normalized

    @staticmethod
    def _occurrence_overlaps_chunk(
        occurrence: dict[str, Any],
        *,
        start_offset: int,
        end_offset: int,
    ) -> bool:
        """Return whether an image occurrence overlaps the current chunk window."""

        occurrence_start = int(occurrence.get("text_offset", 0))
        occurrence_end = occurrence_start + int(occurrence.get("text_length", 0))
        return occurrence_start < end_offset and occurrence_end > start_offset

    @staticmethod
    def _merge_image_refs(*groups: list[str]) -> list[str]:
        """Merge image ids while preserving first-seen order."""

        ordered: list[str] = []
        seen: set[str] = set()
        for group in groups:
            for raw_ref in group:
                image_id = str(raw_ref).strip()
                if not image_id or image_id in seen:
                    continue
                seen.add(image_id)
                ordered.append(image_id)
        return ordered

    @staticmethod
    def _build_chunk_metadata(
        *,
        document_metadata: dict[str, Any],
        chunk_metadata: dict[str, Any],
        chunk_index: int,
        source_ref: str | None,
        image_refs: list[str],
        image_occurrences: list[dict[str, Any]],
        images: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Merge document and chunk metadata while keeping only chunk-scoped image fields."""

        normalized_metadata = {
            key: value
            for key, value in document_metadata.items()
            if key not in {"images", "image_occurrences"}
        }
        normalized_metadata.update(
            {
                key: value
                for key, value in chunk_metadata.items()
                if key not in {"images", "image_occurrences", "image_refs", "chunk_index", "source_ref"}
            }
        )
        normalized_metadata["chunk_index"] = chunk_index
        normalized_metadata["source_ref"] = source_ref
        normalized_metadata["image_refs"] = list(image_refs)
        if image_occurrences:
            normalized_metadata["image_occurrences"] = [dict(item) for item in image_occurrences]
        if images:
            normalized_metadata["images"] = [dict(item) for item in images]
        return normalized_metadata
