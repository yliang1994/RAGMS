"""Management service that exposes ingestion actions to dashboard pages."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ragms.ingestion_pipeline import PipelineCallback, ProgressEvent
from ragms.ingestion_pipeline.bootstrap import build_ingestion_pipeline, run_ingestion_batch
from ragms.ingestion_pipeline.lifecycle import DocumentRegistry, IngestionDocumentManager
from ragms.runtime.settings_models import AppSettings
from ragms.storage.sqlite.connection import create_sqlite_connection
from ragms.storage.sqlite.repositories import (
    DocumentsRepository,
    ImagesRepository,
    IngestionHistoryRepository,
)


class DocumentAdminService:
    """Provide dashboard-safe document ingestion, rebuild, and delete actions."""

    def __init__(
        self,
        settings: AppSettings,
        *,
        connection: sqlite3.Connection | None = None,
        pipeline_factory: Callable[..., Any] | None = None,
        batch_runner: Callable[..., list[dict[str, Any]]] | None = None,
    ) -> None:
        self.settings = settings
        self.connection = connection or create_sqlite_connection(settings.storage.sqlite.path)
        self.documents_repository = DocumentsRepository(self.connection)
        self.images_repository = ImagesRepository(self.connection)
        self.ingestion_history = IngestionHistoryRepository(self.connection)
        self.pipeline_factory = pipeline_factory or build_ingestion_pipeline
        self.batch_runner = batch_runner or run_ingestion_batch
        self._bm25_index_dir = settings.paths.data_dir / "indexes" / "sparse"
        self._upload_root = settings.paths.data_dir / "dashboard_uploads"

    def ingest_documents(
        self,
        sources: Sequence[str | Path],
        *,
        collection: str | None = None,
        force_rebuild: bool = False,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Ingest a batch of sources through the service-layer pipeline entrypoint."""

        callback = _DashboardProgressCallback(on_progress)
        resolved_collection = collection or self.settings.vector_store.collection
        pipeline = self.pipeline_factory(
            self.settings,
            collection=resolved_collection,
            callbacks=[callback],
        )
        results = self.batch_runner(
            pipeline,
            sources=[str(Path(source)) for source in sources],
            collection=resolved_collection,
            force_rebuild=force_rebuild,
        )
        return {
            "collection": resolved_collection,
            "force_rebuild": force_rebuild,
            "results": results,
            "progress_events": callback.events,
        }

    def delete_document(self, document_id: str) -> dict[str, Any]:
        """Delete a document and clear associated local management state."""

        manager = self._build_lifecycle_manager()
        return manager.delete(document_id)

    def rebuild_document(self, document_id: str) -> dict[str, Any]:
        """Reset a document for re-ingestion and clear stale local state."""

        manager = self._build_lifecycle_manager()
        return manager.rebuild(document_id)

    def save_uploads(
        self,
        uploads: Sequence[dict[str, Any]],
    ) -> list[str]:
        """Persist uploaded dashboard files into a controlled temp directory."""

        self._upload_root.mkdir(parents=True, exist_ok=True)
        saved_paths: list[str] = []
        for item in uploads:
            name = str(item.get("name") or "upload.bin").strip() or "upload.bin"
            content = item.get("content", b"")
            target = self._upload_root / name
            if isinstance(content, str):
                data = content.encode("utf-8")
            else:
                data = bytes(content)
            target.write_bytes(data)
            saved_paths.append(str(target))
        return saved_paths

    def _build_lifecycle_manager(self) -> IngestionDocumentManager:
        registry = DocumentRegistry(self.documents_repository)
        return IngestionDocumentManager(
            registry=registry,
            ingestion_history=self.ingestion_history,
            vector_cleanup=self._delete_vector_entries,
            bm25_cleanup=self._delete_bm25_entries,
            image_cleanup=self._delete_image_entries,
        )

    def _delete_vector_entries(self, record: dict[str, object]) -> int:
        """Return zero until vector-store delete-by-document is introduced."""

        return 0

    def _delete_bm25_entries(self, record: dict[str, object]) -> int:
        """Delete snapshot entries matching one document from all sparse indexes."""

        document_id = str(record["document_id"])
        source_path = str(record["source_path"])
        removed = 0
        if not self._bm25_index_dir.is_dir():
            return removed

        for path in self._bm25_index_dir.glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            documents = dict(payload.get("documents") or {})
            retained: dict[str, Any] = {}
            for chunk_id, chunk in documents.items():
                if str(chunk.get("document_id") or "") == document_id or str(chunk.get("source_path") or "") == source_path:
                    removed += 1
                    continue
                retained[chunk_id] = chunk
            if retained != documents:
                payload["documents"] = retained
                path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return removed

    def _delete_image_entries(self, record: dict[str, object]) -> int:
        """Delete image rows and persisted files for one document."""

        document_id = str(record["document_id"])
        image_rows = self.images_repository.list_by_document_id(document_id)
        for row in image_rows:
            file_path = Path(str(row.get("file_path") or "")).expanduser()
            if file_path.is_file():
                file_path.unlink()
        self.images_repository.delete_by_document_id(document_id)
        return len(image_rows)


class _DashboardProgressCallback(PipelineCallback):
    """Capture normalized progress events for dashboard rendering and tests."""

    def __init__(self, callback: Callable[[dict[str, Any]], None] | None) -> None:
        self.callback = callback
        self.events: list[dict[str, Any]] = []

    def on_progress(self, event: ProgressEvent) -> None:
        payload = {
            "trace_id": event.trace_id,
            "source_path": event.source_path,
            "document_id": event.document_id,
            "completed_stages": event.completed_stages,
            "total_stages": event.total_stages,
            "current_stage": event.current_stage,
            "status": event.status,
            "elapsed_ms": event.elapsed_ms,
            "metadata": dict(event.metadata or {}),
        }
        self.events.append(payload)
        if self.callback is not None:
            self.callback(payload)
