"""Reusable ingestion bootstrap helpers shared by CLI and MCP layers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ragms.ingestion_pipeline import IngestionPipeline, PipelineCallback
from ragms.ingestion_pipeline.chunking import ChunkingPipeline
from ragms.ingestion_pipeline.embedding import DenseEncoder, SparseEncoder
from ragms.ingestion_pipeline.file_integrity import FileIntegrity
from ragms.ingestion_pipeline.lifecycle import DocumentRegistry
from ragms.ingestion_pipeline.storage import StoragePipeline, VectorUpsert
from ragms.ingestion_pipeline.storage.bm25_indexing import BM25StorageWriter
from ragms.ingestion_pipeline.storage.image_persistence import ImageStorageWriter
from ragms.ingestion_pipeline.transform import (
    ImageCaptionInjector,
    SemanticMetadataInjector,
    SmartChunkBuilder,
    TransformPipeline,
)
from ragms.ingestion_pipeline.transform.services import MetadataService
from ragms.libs.factories import LLMFactory
from ragms.runtime.container import build_container
from ragms.runtime.settings_models import AppSettings, LLMOverrideSettings
from ragms.storage.images import ImageStorage
from ragms.storage.indexes import BM25Indexer
from ragms.storage.sqlite.repositories import (
    DocumentsRepository,
    ImagesRepository,
    IngestionHistoryRepository,
    ProcessingCacheRepository,
)
from ragms.storage.sqlite.schema import initialize_metadata_schema


def _apply_collection_override(settings: AppSettings, collection: str | None) -> AppSettings:
    """Return settings with an optional collection override applied."""

    if not collection:
        return settings
    updated = settings.model_copy(deep=True)
    updated.vector_store.collection = collection
    return updated


def _resolve_transform_llm(
    settings: AppSettings,
    *,
    override: LLMOverrideSettings,
):
    """Build a transform-scoped LLM by overlaying optional overrides on the primary config."""

    llm_settings = settings.llm.model_copy(
        update=override.model_dump(mode="python", exclude_none=True)
    )
    if not getattr(llm_settings, "api_key", None):
        return None
    return LLMFactory.create(llm_settings)


def build_ingestion_pipeline(
    settings: AppSettings,
    *,
    collection: str | None = None,
    callbacks: list[PipelineCallback] | None = None,
) -> IngestionPipeline:
    """Assemble the end-to-end ingestion pipeline for CLI and MCP execution."""

    resolved_settings = _apply_collection_override(settings, collection)
    container = build_container(resolved_settings)
    connection = initialize_metadata_schema(resolved_settings.storage.sqlite.path)
    file_integrity = FileIntegrity(IngestionHistoryRepository(connection))
    document_registry = DocumentRegistry(DocumentsRepository(connection))
    processing_cache = ProcessingCacheRepository(connection)
    images_repository = ImagesRepository(connection)
    collection_name = resolved_settings.vector_store.collection
    llm = container.get("llm")
    if not getattr(resolved_settings.llm, "api_key", None):
        llm = None
    chunk_refine_llm = _resolve_transform_llm(
        resolved_settings,
        override=resolved_settings.ingestion.transform.chunk_refine_llm,
    )
    metadata_enrich_llm = _resolve_transform_llm(
        resolved_settings,
        override=resolved_settings.ingestion.transform.metadata_enrich_llm,
    )
    vision_llm = container.get("vision_llm")
    if not getattr(resolved_settings.vision_llm, "api_key", None):
        vision_llm = None

    transform = TransformPipeline(
        smart_chunk_builder=SmartChunkBuilder(
            enable_llm_refine=resolved_settings.ingestion.transform.enable_llm_chunk_refine,
            llm=chunk_refine_llm or llm,
            llm_model=(
                getattr(chunk_refine_llm, "model", None)
                if chunk_refine_llm is not None
                else getattr(llm, "model", None) if llm is not None else None
            ),
        ),
        metadata_injector=SemanticMetadataInjector(
            service=MetadataService(
                enable_llm_enrich=resolved_settings.ingestion.transform.enable_llm_metadata_enrich,
                llm=metadata_enrich_llm or llm,
                llm_model=(
                    getattr(metadata_enrich_llm, "model", None)
                    if metadata_enrich_llm is not None
                    else getattr(llm, "model", None) if llm is not None else None
                ),
            ),
            enable_llm_enrich=resolved_settings.ingestion.transform.enable_llm_metadata_enrich,
        ),
        image_captioner=ImageCaptionInjector(
            vision_llm=vision_llm,
            cache_repository=processing_cache,
            model=getattr(vision_llm, "model", None) if vision_llm is not None else None,
        ),
        fail_open=True,
    )
    return IngestionPipeline(
        loader=container.get("loader"),
        splitter=container.get("splitter"),
        transform=transform,
        embedding=container.get("embedding"),
        vector_store=container.get("vector_store"),
        file_integrity=file_integrity,
        document_registry=document_registry,
        callbacks=callbacks,
        chunking_pipeline=ChunkingPipeline(container.get("splitter")),
        dense_encoder=DenseEncoder(
            container.get("embedding"),
            batch_size=resolved_settings.embedding.batch_size,
        ),
        sparse_encoder=SparseEncoder(),
        storage_pipeline=StoragePipeline(
            vector_upsert=VectorUpsert(container.get("vector_store"))
        ),
        bm25_writer=BM25StorageWriter(
            BM25Indexer(
                index_dir=resolved_settings.paths.data_dir / "indexes" / "sparse",
                collection=collection_name,
            )
        ),
        image_storage_writer=ImageStorageWriter(
            image_storage=ImageStorage(root_dir=resolved_settings.paths.data_dir / "images"),
            repository=images_repository,
            collection=collection_name,
        ),
    )


def _discover_sources(path: str | Path) -> list[Path]:
    """Resolve a file or directory into a stable list of source files."""

    target = Path(path).expanduser().resolve()
    if target.is_file():
        return [target]
    if target.is_dir():
        return sorted(item for item in target.rglob("*") if item.is_file())
    raise FileNotFoundError(f"Ingestion path does not exist: {target}")


def discover_ingestion_sources(paths: Sequence[str | Path]) -> tuple[list[Path], list[dict[str, str]]]:
    """Resolve one or more inputs into a deduplicated list of source files and path errors."""

    discovered: list[Path] = []
    seen: set[str] = set()
    errors: list[dict[str, str]] = []

    for raw_path in paths:
        try:
            sources = _discover_sources(raw_path)
        except FileNotFoundError as exc:
            errors.append(
                {
                    "path": str(Path(raw_path).expanduser()),
                    "message": str(exc),
                }
            )
            continue

        for source in sources:
            normalized = str(source)
            if normalized in seen:
                continue
            seen.add(normalized)
            discovered.append(source)

    return discovered, errors


def run_ingestion_batch(
    pipeline: IngestionPipeline,
    *,
    sources: Sequence[str | Path],
    collection: str,
    force_rebuild: bool,
) -> list[dict[str, Any]]:
    """Execute ingestion for a list of resolved sources and keep the raw pipeline payloads."""

    results: list[dict[str, Any]] = []
    for source in sources:
        result = pipeline.run(
            source,
            metadata={
                "collection": collection,
                "force_rebuild": force_rebuild,
            },
        )
        results.append(
            {
                "source_path": str(Path(source)),
                "result": result,
            }
        )
    return results
