from __future__ import annotations

import hashlib
from pathlib import Path

from ragms.ingestion_pipeline.file_integrity import FileIntegrity
from ragms.storage.sqlite.schema import initialize_metadata_schema
from ragms.storage.sqlite.repositories import IngestionHistoryRepository


def test_file_integrity_computes_sha256_for_source_file(tmp_path: Path) -> None:
    source_path = tmp_path / "sample.txt"
    source_path.write_text("hello ragms", encoding="utf-8")
    repository = IngestionHistoryRepository(initialize_metadata_schema(tmp_path / "ragms.db"))
    integrity = FileIntegrity(repository)

    digest = integrity.compute_sha256(source_path)

    assert digest == hashlib.sha256(b"hello ragms").hexdigest()


def test_file_integrity_skips_unchanged_content_once_it_is_recorded(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.db"
    source_path = tmp_path / "sample.txt"
    source_path.write_text("same content", encoding="utf-8")

    repository = IngestionHistoryRepository(initialize_metadata_schema(database_path))
    integrity = FileIntegrity(repository)
    digest = integrity.compute_sha256(source_path)

    assert integrity.should_skip(source_path) is False

    stored = repository.record_processed(
        source_path=str(source_path),
        source_sha256=digest,
        document_id="doc-1",
    )

    assert stored["source_sha256"] == digest
    assert integrity.should_skip(source_path) is True


def test_file_integrity_detects_content_changes_and_honors_force_rebuild(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.db"
    source_path = tmp_path / "sample.txt"
    source_path.write_text("version-one", encoding="utf-8")

    repository = IngestionHistoryRepository(initialize_metadata_schema(database_path))
    integrity = FileIntegrity(repository)
    repository.record_processed(
        source_path=str(source_path),
        source_sha256=integrity.compute_sha256(source_path),
        document_id="doc-1",
    )

    source_path.write_text("version-two", encoding="utf-8")

    assert integrity.should_skip(source_path) is False
    assert integrity.should_skip(source_path, force_rebuild=True) is False
