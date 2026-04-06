"""Minimal local ingestion bootstrap."""

from __future__ import annotations

import argparse
from pathlib import Path
from collections.abc import Sequence

from ragms.ingestion_pipeline import IngestionPipeline, PipelineCallback, PipelineEvent, StageEvent, ErrorEvent
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
from ragms.runtime.config import load_settings
from ragms.runtime.container import build_container
from ragms.runtime.settings_models import AppSettings
from ragms.storage.images import ImageStorage
from ragms.storage.indexes import BM25Indexer
from ragms.storage.sqlite.repositories import (
    DocumentsRepository,
    ImagesRepository,
    IngestionHistoryRepository,
    ProcessingCacheRepository,
)
from ragms.storage.sqlite.schema import initialize_metadata_schema


class ConsoleProgressCallback(PipelineCallback):
    """Print lightweight stage progress for local CLI usage."""

    def on_pipeline_start(self, event: PipelineEvent) -> None:
        print(f"pipeline trace_id={event.trace_id} source={event.source_path} status={event.status}")

    def on_stage_start(self, event: StageEvent) -> None:
        print(f"stage={event.stage} status=started source={event.source_path}")

    def on_stage_end(self, event: StageEvent) -> None:
        print(f"stage={event.stage} status={event.status} elapsed_ms={event.elapsed_ms}")

    def on_error(self, event: ErrorEvent) -> None:
        print(
            f"stage={event.stage} status=failed source={event.source_path} "
            f"error={event.error.get('message', 'unknown')}"
        )


def _apply_collection_override(settings: AppSettings, collection: str | None) -> AppSettings:
    """Return settings with an optional collection override applied."""

    if not collection:
        return settings
    updated = settings.model_copy(deep=True)
    updated.vector_store.collection = collection
    return updated


def build_ingestion_pipeline(
    settings: AppSettings,
    *,
    collection: str | None = None,
    callbacks: list[PipelineCallback] | None = None,
) -> IngestionPipeline:
    """Assemble the end-to-end ingestion pipeline for CLI execution."""

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
    vision_llm = container.get("vision_llm")
    if not getattr(resolved_settings.vision_llm, "api_key", None):
        vision_llm = None

    transform = TransformPipeline(
        smart_chunk_builder=SmartChunkBuilder(
            enable_llm_refine=resolved_settings.ingestion.transform.enable_llm_chunk_refine,
            llm=llm,
            llm_model=getattr(llm, "model", None) if llm is not None else None,
        ),
        metadata_injector=SemanticMetadataInjector(
            service=MetadataService(
                enable_llm_enrich=resolved_settings.ingestion.transform.enable_llm_metadata_enrich,
                llm=llm,
                llm_model=getattr(llm, "model", None) if llm is not None else None,
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


def ingest_documents_main(argv: Sequence[str] | None = None) -> int:
    """Run local ingestion for one file or a directory tree."""

    parser = argparse.ArgumentParser(description="Run the local RagMS ingestion bootstrap.")
    parser.add_argument(
        "--settings",
        default="settings.yaml",
        help="Path to the settings.yaml file.",
    )
    parser.add_argument(
        "--path",
        default="data/raw/documents",
        help="File or directory to ingest.",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Optional collection override for vector, BM25, and image storage.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force ingestion even when the content hash was already indexed.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    settings = load_settings(args.settings)
    pipeline = build_ingestion_pipeline(
        settings,
        collection=args.collection,
        callbacks=[ConsoleProgressCallback()],
    )
    sources = _discover_sources(args.path)
    exit_code = 0
    for source in sources:
        print(f"ingest source={source} force={str(args.force).lower()}")
        result = pipeline.run(
            source,
            metadata={
                "collection": args.collection or settings.vector_store.collection,
                "force_rebuild": args.force,
            },
        )
        final_status = str(result.get("lifecycle", {}).get("final_status") or result.get("status"))
        print(
            f"result source={source} status={final_status} "
            f"chunks={len(result.get('smart_chunks', []))} "
            f"stored={len(result.get('stored_ids', []))}"
        )
        if result.get("status") == "failed":
            exit_code = 1
            print(f"failure source={source} error={result.get('error', {}).get('message', 'unknown')}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(ingest_documents_main())
