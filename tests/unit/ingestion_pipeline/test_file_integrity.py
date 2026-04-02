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

    stored = integrity.mark_success(source_path, source_sha256=digest, document_id="doc-1")

    assert stored["source_sha256"] == digest
    assert stored["status"] == "indexed"
    assert stored["last_error"] is None
    assert integrity.should_skip(source_path) is True


def test_file_integrity_detects_content_changes_and_honors_force_rebuild(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.db"
    source_path = tmp_path / "sample.txt"
    source_path.write_text("version-one", encoding="utf-8")

    repository = IngestionHistoryRepository(initialize_metadata_schema(database_path))
    integrity = FileIntegrity(repository)
    integrity.mark_success(source_path, document_id="doc-1")

    source_path.write_text("version-two", encoding="utf-8")

    assert integrity.should_skip(source_path) is False
    assert integrity.should_skip(source_path, force_rebuild=True) is False


def test_file_integrity_failed_records_do_not_trigger_skip_and_preserve_error(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.db"
    source_path = tmp_path / "sample.txt"
    source_path.write_text("same content", encoding="utf-8")

    repository = IngestionHistoryRepository(initialize_metadata_schema(database_path))
    integrity = FileIntegrity(repository)

    failed = integrity.mark_failed(
        source_path,
        error_message="ocr timeout",
        document_id="doc-1",
        config_version="v2",
    )

    assert failed["status"] == "failed"
    assert failed["last_error"] == "ocr timeout"
    assert failed["config_version"] == "v2"
    assert failed["started_at"] is not None
    assert integrity.should_skip(source_path) is False


def test_file_integrity_success_updates_source_path_for_same_content_hash(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.db"
    original_path = tmp_path / "docs" / "sample.txt"
    renamed_path = tmp_path / "renamed" / "sample.txt"
    original_path.parent.mkdir(parents=True, exist_ok=True)
    renamed_path.parent.mkdir(parents=True, exist_ok=True)
    original_path.write_text("same content", encoding="utf-8")
    renamed_path.write_text("same content", encoding="utf-8")

    repository = IngestionHistoryRepository(initialize_metadata_schema(database_path))
    integrity = FileIntegrity(repository)

    first = integrity.mark_success(original_path, document_id="doc-1")
    second = integrity.mark_success(renamed_path, document_id="doc-1")

    assert first["source_sha256"] == second["source_sha256"]
    assert second["source_path"] == str(renamed_path)
    row_count = repository.connection.execute(
        "SELECT COUNT(*) AS count FROM ingestion_history"
    ).fetchone()["count"]
    assert row_count == 1
