from __future__ import annotations

from ragms.ingestion_pipeline.lifecycle.document_registry import DocumentRegistry
from ragms.storage.sqlite.connection import create_sqlite_connection
from ragms.storage.sqlite.schema import initialize_metadata_schema
from ragms.storage.sqlite.repositories import DocumentsRepository


def test_document_registry_persists_across_connection_restarts(tmp_path) -> None:
    database_path = tmp_path / "metadata.db"

    first_connection = initialize_metadata_schema(database_path)
    first_registry = DocumentRegistry(DocumentsRepository(first_connection))
    first_registry.register(
        source_path="docs/a.pdf",
        source_sha256="sha-1",
        document_id="doc-1",
    )
    first_registry.update_status("doc-1", status="processing", current_stage="split")
    first_connection.close()

    second_connection = create_sqlite_connection(database_path)
    second_registry = DocumentRegistry(DocumentsRepository(second_connection))
    restored = second_registry.get("doc-1")

    assert restored is not None
    assert restored["status"] == "processing"
    assert restored["current_stage"] == "split"
    assert second_registry.find_by_source_path("docs/a.pdf")["document_id"] == "doc-1"
