"""Semantic metadata injection for smart chunks."""

from __future__ import annotations

from typing import Any

from ragms.libs.abstractions import BaseTransform

from .services.metadata_service import MetadataService


def inject_semantic_metadata(
    chunks: list[dict[str, Any]],
    *,
    service: MetadataService | None = None,
    enable_llm_enrich: bool | None = None,
    context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Inject semantic metadata into chunks with the default metadata service."""

    injector = SemanticMetadataInjector(service=service, enable_llm_enrich=enable_llm_enrich)
    return injector.transform(chunks, context=context)


class SemanticMetadataInjector(BaseTransform):
    """Attach deterministic semantic metadata onto chunks."""

    def __init__(
        self,
        *,
        service: MetadataService | None = None,
        enable_llm_enrich: bool | None = None,
    ) -> None:
        self.service = service or MetadataService()
        self.enable_llm_enrich = (
            self.service.enable_llm_enrich if enable_llm_enrich is None else enable_llm_enrich
        )

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
            metadata["metadata_enriched_by"] = "rule"
            metadata.pop("fallback_reason", None)

            if self.enable_llm_enrich:
                try:
                    semantic = self.service.enrich_with_llm(
                        item,
                        base_semantic=semantic,
                        context=context,
                    )
                except Exception as exc:
                    metadata["fallback_reason"] = str(exc)
                else:
                    metadata["metadata_enriched_by"] = "llm"

            metadata["semantic"] = semantic
            metadata["chunk_title"] = semantic["title"]
            metadata["chunk_summary"] = semantic["summary"]
            metadata["chunk_tags"] = list(semantic["tags"])
            if self.enable_llm_enrich:
                metadata["prompt_version"] = self.service.llm_prompt_version
                model_name = self.service.llm_model or getattr(self.service._llm, "model", None)
                if model_name:
                    metadata["model"] = str(model_name)
            item["metadata"] = metadata
            item["content"] = original_content
            enriched_chunks.append(item)
        return enriched_chunks
