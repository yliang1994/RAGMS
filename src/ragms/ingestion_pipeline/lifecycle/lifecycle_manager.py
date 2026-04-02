"""Document lifecycle orchestration for delete and rebuild operations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ragms.ingestion_pipeline.lifecycle.document_registry import DocumentRegistry, DocumentRegistryError
from ragms.storage.sqlite.repositories import IngestionHistoryRepository


class IngestionLifecycleError(ValueError):
    """Raised when a lifecycle operation cannot be completed."""


CleanupHook = Callable[[dict[str, object]], int]


class IngestionDocumentManager:
    """Coordinate document rebuild and delete operations across local state stores."""

    def __init__(
        self,
        *,
        registry: DocumentRegistry,
        ingestion_history: IngestionHistoryRepository,
        vector_cleanup: CleanupHook | None = None,
        image_cleanup: CleanupHook | None = None,
    ) -> None:
        self.registry = registry
        self.ingestion_history = ingestion_history
        self.vector_cleanup = vector_cleanup
        self.image_cleanup = image_cleanup

    def rebuild(self, document_id: str) -> dict[str, object]:
        """Prepare a document for re-ingestion and clear stale persisted state."""

        record = self._require_document(document_id)
        vector_entries = self._run_cleanup(self.vector_cleanup, record)
        image_entries = self._run_cleanup(self.image_cleanup, record)
        deleted_history = self.ingestion_history.delete_by_document_id(document_id)
        updated = self.registry.register(
            document_id=document_id,
            source_path=str(record["source_path"]),
            source_sha256=str(record["source_sha256"]),
            status="pending",
            current_stage="rebuild_requested",
        )
        return {
            "action": "rebuild",
            "document": updated,
            "cleanup": {
                "ingestion_history": deleted_history,
                "vector_entries": vector_entries,
                "image_entries": image_entries,
            },
        }

    def delete(self, document_id: str) -> dict[str, object]:
        """Mark a document deleted and clear associated persisted state."""

        record = self._require_document(document_id)
        vector_entries = self._run_cleanup(self.vector_cleanup, record)
        image_entries = self._run_cleanup(self.image_cleanup, record)
        deleted_history = self.ingestion_history.delete_by_document_id(document_id)
        if str(record["status"]) == "deleted":
            updated = record
        else:
            updated = self.registry.update_status(
                document_id,
                status="deleted",
                current_stage="deleted",
            )
        return {
            "action": "delete",
            "document": updated,
            "cleanup": {
                "ingestion_history": deleted_history,
                "vector_entries": vector_entries,
                "image_entries": image_entries,
            },
        }

    def _require_document(self, document_id: str) -> dict[str, object]:
        """Load a document record or raise a lifecycle-specific error."""

        record = self.registry.get(document_id)
        if record is None:
            raise IngestionLifecycleError(f"Unknown document: {document_id}")
        return record

    @staticmethod
    def _run_cleanup(hook: CleanupHook | None, record: dict[str, object]) -> int:
        """Execute an optional cleanup hook and normalize its return value."""

        if hook is None:
            return 0
        return int(hook(record))
