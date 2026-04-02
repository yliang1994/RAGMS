"""Semantic metadata injection for smart chunks."""

from __future__ import annotations

from typing import Any

from ragms.libs.abstractions import BaseTransform

from .services.metadata_service import MetadataService


def inject_semantic_metadata(
    chunks: list[dict[str, Any]],
    *,
    service: MetadataService | None = None,
    context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Inject semantic metadata into chunks with the default metadata service."""

    injector = SemanticMetadataInjector(service=service)
    return injector.transform(chunks, context=context)


class SemanticMetadataInjector(BaseTransform):
    """Attach deterministic semantic metadata onto chunks."""

    def __init__(self, *, service: MetadataService | None = None) -> None:
        self.service = service or MetadataService()

    def transform(
        self,
        chunks: list[dict[str, Any]],
        *,
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Enrich every chunk in order and preserve chunk shape."""

        enriched_chunks: list[dict[str, Any]] = []
        for chunk in chunks:
            item = dict(chunk)
            original_content = item.get("content")
            metadata = dict(item.get("metadata") or {})
            semantic = self.service.enrich(item, context=context)
            metadata["semantic"] = semantic
            metadata["chunk_title"] = semantic["title"]
            metadata["chunk_summary"] = semantic["summary"]
            metadata["chunk_tags"] = list(semantic["tags"])
            metadata["metadata_enriched_by"] = "rule"
            item["metadata"] = metadata
            item["content"] = original_content
            enriched_chunks.append(item)
        return enriched_chunks
