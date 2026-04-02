from __future__ import annotations

import pytest

from ragms.ingestion_pipeline.lifecycle import (
    DocumentRegistry,
    IngestionDocumentManager,
    IngestionLifecycleError,
)
from ragms.storage.sqlite.repositories import DocumentsRepository, IngestionHistoryRepository
from ragms.storage.sqlite.schema import initialize_metadata_schema


class RecordingCleanup:
    def __init__(self, deleted_count: int) -> None:
        self.deleted_count = deleted_count
        self.calls: list[dict[str, object]] = []

    def __call__(self, document: dict[str, object]) -> int:
        self.calls.append(document)
        return self.deleted_count


class ExplodingCleanup:
    def __call__(self, document: dict[str, object]) -> int:
        del document
        raise RuntimeError("cleanup failed")


def _build_manager(tmp_path):
    connection = initialize_metadata_schema(tmp_path / "metadata.db")
    registry = DocumentRegistry(DocumentsRepository(connection))
    history = IngestionHistoryRepository(connection)
    vector_cleanup = RecordingCleanup(2)
    bm25_cleanup = RecordingCleanup(3)
    image_cleanup = RecordingCleanup(1)
    manager = IngestionDocumentManager(
        registry=registry,
        ingestion_history=history,
        vector_cleanup=vector_cleanup,
        bm25_cleanup=bm25_cleanup,
        image_cleanup=image_cleanup,
    )
    return manager, registry, history, vector_cleanup, bm25_cleanup, image_cleanup


def test_lifecycle_manager_delete_marks_document_deleted_and_cleans_state(tmp_path) -> None:
    manager, registry, history, vector_cleanup, bm25_cleanup, image_cleanup = _build_manager(tmp_path)
    registry.register(source_path="docs/a.pdf", source_sha256="sha-1", document_id="doc-1")
    registry.update_status("doc-1", status="processing", current_stage="embed")
    history.record_processed(source_path="docs/a.pdf", source_sha256="sha-1", document_id="doc-1")

    result = manager.delete("doc-1")

    assert result["action"] == "delete"
    assert result["document"]["status"] == "deleted"
    assert result["document"]["current_stage"] == "deleted"
    assert result["cleanup"] == {
        "ingestion_history": 1,
        "vector_entries": 2,
        "bm25_entries": 3,
        "image_entries": 1,
    }
    assert history.get_by_sha256("sha-1") is None
    assert len(vector_cleanup.calls) == 1
    assert len(bm25_cleanup.calls) == 1
    assert len(image_cleanup.calls) == 1


def test_lifecycle_manager_rebuild_resets_document_to_pending_and_cleans_state(tmp_path) -> None:
    manager, registry, history, vector_cleanup, bm25_cleanup, image_cleanup = _build_manager(tmp_path)
    registry.register(source_path="docs/a.pdf", source_sha256="sha-1", document_id="doc-1")
    registry.update_status("doc-1", status="processing", current_stage="store")
    registry.update_status("doc-1", status="indexed", current_stage="store")
    history.record_processed(source_path="docs/a.pdf", source_sha256="sha-1", document_id="doc-1")

    result = manager.rebuild("doc-1")

    assert result["action"] == "rebuild"
    assert result["document"]["status"] == "pending"
    assert result["document"]["current_stage"] == "rebuild_requested"
    assert result["cleanup"] == {
        "ingestion_history": 1,
        "vector_entries": 2,
        "bm25_entries": 3,
        "image_entries": 1,
    }
    assert history.get_by_sha256("sha-1") is None
    assert len(vector_cleanup.calls) == 1
    assert len(bm25_cleanup.calls) == 1
    assert len(image_cleanup.calls) == 1


def test_lifecycle_manager_can_rebuild_a_deleted_document(tmp_path) -> None:
    manager, registry, history, _vector_cleanup, _bm25_cleanup, _image_cleanup = _build_manager(tmp_path)
    registry.register(source_path="docs/a.pdf", source_sha256="sha-1", document_id="doc-1")
    registry.update_status("doc-1", status="processing", current_stage="load")
    registry.update_status("doc-1", status="failed", current_stage="load")
    manager.delete("doc-1")

    result = manager.rebuild("doc-1")

    assert result["document"]["status"] == "pending"
    assert result["document"]["current_stage"] == "rebuild_requested"


def test_lifecycle_manager_delete_is_idempotent_for_already_deleted_documents(tmp_path) -> None:
    manager, registry, history, _vector_cleanup, _bm25_cleanup, _image_cleanup = _build_manager(tmp_path)
    registry.register(source_path="docs/a.pdf", source_sha256="sha-1", document_id="doc-1")

    first = manager.delete("doc-1")
    second = manager.delete("doc-1")

    assert first["document"]["status"] == "deleted"
    assert second["document"]["status"] == "deleted"
    assert second["cleanup"]["ingestion_history"] == 0
    assert history.get_by_sha256("sha-1") is None


def test_lifecycle_manager_wraps_cleanup_failures_as_domain_errors(tmp_path) -> None:
    connection = initialize_metadata_schema(tmp_path / "metadata.db")
    registry = DocumentRegistry(DocumentsRepository(connection))
    history = IngestionHistoryRepository(connection)
    manager = IngestionDocumentManager(
        registry=registry,
        ingestion_history=history,
        vector_cleanup=ExplodingCleanup(),
    )
    registry.register(source_path="docs/a.pdf", source_sha256="sha-1", document_id="doc-1")

    with pytest.raises(IngestionLifecycleError, match="Lifecycle cleanup failed in vector_cleanup"):
        manager.delete("doc-1")


def test_lifecycle_manager_rejects_unknown_documents(tmp_path) -> None:
    manager, _registry, _history, _vector_cleanup, _bm25_cleanup, _image_cleanup = _build_manager(tmp_path)

    with pytest.raises(IngestionLifecycleError, match="Unknown document: missing"):
        manager.delete("missing")

    with pytest.raises(IngestionLifecycleError, match="Unknown document: missing"):
        manager.rebuild("missing")
