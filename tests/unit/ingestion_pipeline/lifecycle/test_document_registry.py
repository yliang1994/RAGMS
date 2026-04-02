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
    assert record["failure_reason"] is None
    assert record["last_ingested_at"] is None
    assert record["version"] == 1
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
    assert processing["failure_reason"] is None
    assert indexed["status"] == "indexed"
    assert indexed["current_stage"] == "store"
    assert indexed["last_ingested_at"] is not None


def test_document_registry_tracks_failure_reason_and_clears_it_after_recovery(tmp_path) -> None:
    repository = DocumentsRepository(initialize_metadata_schema(tmp_path / "metadata.db"))
    registry = DocumentRegistry(repository)
    registry.register(source_path="docs/a.pdf", source_sha256="sha-1", document_id="doc-1")

    registry.update_status("doc-1", status="processing", current_stage="load")
    failed = registry.update_status(
        "doc-1",
        status="failed",
        current_stage="transform",
        failure_reason="transform blew up",
    )
    recovered = registry.update_status("doc-1", status="processing", current_stage="retry")

    assert failed["failure_reason"] == "transform blew up"
    assert failed["current_stage"] == "transform"
    assert recovered["failure_reason"] is None
    assert recovered["current_stage"] == "retry"


def test_document_registry_builds_stable_document_id_and_increments_version_on_new_content(tmp_path) -> None:
    repository = DocumentsRepository(initialize_metadata_schema(tmp_path / "metadata.db"))
    registry = DocumentRegistry(repository)

    first = registry.register(source_path="docs/a.pdf", source_sha256="sha-1")
    second = registry.register(source_path="docs/renamed-a.pdf", source_sha256="sha-1")
    third = registry.register(
        source_path="docs/renamed-a.pdf",
        source_sha256="sha-2",
        document_id=str(first["document_id"]),
    )

    assert first["document_id"] == second["document_id"]
    assert first["version"] == 1
    assert second["version"] == 1
    assert third["document_id"] == first["document_id"]
    assert third["version"] == 2


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
