from __future__ import annotations

import pytest

from ragms.ingestion_pipeline.lifecycle.document_registry import (
    DocumentRegistry,
    DocumentRegistryError,
)
from ragms.storage.sqlite.schema import initialize_metadata_schema
from ragms.storage.sqlite.repositories import DocumentsRepository


def test_document_registry_registers_and_queries_document(tmp_path) -> None:
    repository = DocumentsRepository(initialize_metadata_schema(tmp_path / "metadata.db"))
    registry = DocumentRegistry(repository)

    record = registry.register(
        source_path="docs/a.pdf",
        source_sha256="sha-1",
        document_id="doc-1",
    )

    assert record["document_id"] == "doc-1"
    assert record["status"] == "pending"
    assert record["current_stage"] == "registered"
    assert registry.get("doc-1")["source_path"] == "docs/a.pdf"
    assert registry.find_by_source_path("docs/a.pdf")["source_sha256"] == "sha-1"


def test_document_registry_updates_status_and_stage(tmp_path) -> None:
    repository = DocumentsRepository(initialize_metadata_schema(tmp_path / "metadata.db"))
    registry = DocumentRegistry(repository)
    registry.register(source_path="docs/a.pdf", source_sha256="sha-1", document_id="doc-1")

    processing = registry.update_status("doc-1", status="processing", current_stage="load")
    indexed = registry.update_status("doc-1", status="indexed", current_stage="store")

    assert processing["status"] == "processing"
    assert processing["current_stage"] == "load"
    assert indexed["status"] == "indexed"
    assert indexed["current_stage"] == "store"


def test_document_registry_rejects_illegal_status_transition(tmp_path) -> None:
    repository = DocumentsRepository(initialize_metadata_schema(tmp_path / "metadata.db"))
    registry = DocumentRegistry(repository)
    registry.register(source_path="docs/a.pdf", source_sha256="sha-1", document_id="doc-1")

    with pytest.raises(DocumentRegistryError, match="Illegal document status transition"):
        registry.update_status("doc-1", status="indexed")


def test_document_registry_rejects_unknown_document_lookup_and_transition(tmp_path) -> None:
    repository = DocumentsRepository(initialize_metadata_schema(tmp_path / "metadata.db"))
    registry = DocumentRegistry(repository)

    with pytest.raises(DocumentRegistryError, match="Unknown document: missing"):
        registry.update_status("missing", status="processing")
