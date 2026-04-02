from __future__ import annotations

from ragms.ingestion_pipeline.lifecycle import DocumentRegistry, IngestionDocumentManager
from ragms.storage.sqlite.connection import create_sqlite_connection
from ragms.storage.sqlite.repositories import DocumentsRepository, IngestionHistoryRepository
from ragms.storage.sqlite.schema import initialize_metadata_schema


def test_document_lifecycle_delete_and_rebuild_persist_state(tmp_path) -> None:
    database_path = tmp_path / "metadata.db"
    connection = initialize_metadata_schema(database_path)
    registry = DocumentRegistry(DocumentsRepository(connection))
    history = IngestionHistoryRepository(connection)
    manager = IngestionDocumentManager(
        registry=registry,
        ingestion_history=history,
    )

    registry.register(source_path="docs/a.pdf", source_sha256="sha-1", document_id="doc-1")
    registry.update_status("doc-1", status="processing", current_stage="split")
    registry.update_status("doc-1", status="indexed", current_stage="store")
    history.record_processed(source_path="docs/a.pdf", source_sha256="sha-1", document_id="doc-1")

    deleted = manager.delete("doc-1")
    rebuilt = manager.rebuild("doc-1")
    connection.close()

    reopened = create_sqlite_connection(database_path)
    restored_registry = DocumentRegistry(DocumentsRepository(reopened))
    restored_history = IngestionHistoryRepository(reopened)
    restored = restored_registry.get("doc-1")

    assert deleted["document"]["status"] == "deleted"
    assert rebuilt["document"]["status"] == "pending"
    assert rebuilt["document"]["current_stage"] == "rebuild_requested"
    assert restored is not None
    assert restored["status"] == "pending"
    assert restored["current_stage"] == "rebuild_requested"
    assert restored_history.get_by_sha256("sha-1") is None


def test_document_lifecycle_delete_and_rebuild_are_idempotent(tmp_path) -> None:
    database_path = tmp_path / "metadata.db"
    connection = initialize_metadata_schema(database_path)
    registry = DocumentRegistry(DocumentsRepository(connection))
    history = IngestionHistoryRepository(connection)
    manager = IngestionDocumentManager(
        registry=registry,
        ingestion_history=history,
    )

    registry.register(source_path="docs/a.pdf", source_sha256="sha-1", document_id="doc-1")
    history.record_processed(source_path="docs/a.pdf", source_sha256="sha-1", document_id="doc-1")

    first_delete = manager.delete("doc-1")
    second_delete = manager.delete("doc-1")
    first_rebuild = manager.rebuild("doc-1")
    second_rebuild = manager.rebuild("doc-1")

    assert first_delete["document"]["status"] == "deleted"
    assert second_delete["document"]["status"] == "deleted"
    assert second_delete["cleanup"]["ingestion_history"] == 0
    assert first_rebuild["document"]["status"] == "pending"
    assert second_rebuild["document"]["status"] == "pending"
    assert second_rebuild["document"]["current_stage"] == "rebuild_requested"
