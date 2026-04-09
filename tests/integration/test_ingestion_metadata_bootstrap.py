from __future__ import annotations

import sqlite3
import textwrap
from pathlib import Path

from ragms.runtime.config import load_settings
from ragms.storage.sqlite.schema import initialize_metadata_schema, run_sqlite_migrations
from ragms.storage.sqlite.repositories import IngestionHistoryRepository


def write_settings(path: Path, content: str) -> Path:
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")
    return path


def test_initialize_metadata_schema_creates_default_sqlite_db_and_tables(tmp_path: Path) -> None:
    settings_path = write_settings(
        tmp_path / "settings.yaml",
        """
        app_name: ragms
        environment: test
        paths:
          project_root: .
          data_dir: data
          logs_dir: logs
        llm:
          provider: openai
          model: gpt-4.1-mini
        embedding:
          provider: openai
          model: text-embedding-3-small
        vector_store:
          backend: chroma
          collection: default
        storage:
          sqlite:
            path: data/metadata/ragms.db
        retrieval:
          strategy: hybrid
          fusion_algorithm: rrf
          rerank_backend: disabled
        evaluation:
          backends: [custom_metrics]
        observability:
          enabled: true
          log_file: logs/traces.jsonl
          log_level: INFO
        dashboard:
          enabled: true
          port: 8501
          traces_file: logs/traces.jsonl
        """,
    )

    settings = load_settings(settings_path)
    connection = initialize_metadata_schema(settings.storage.sqlite.path)

    assert settings.storage.sqlite.path == (tmp_path / "data/metadata/ragms.db").resolve()
    assert settings.storage.sqlite.path.exists()

    tables = {
        row["name"]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    assert "schema_migrations" in tables
    assert "documents" in tables
    assert "ingestion_history" in tables
    journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
    assert str(journal_mode).lower() == "wal"
    history_columns = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(ingestion_history)").fetchall()
    }
    assert {"last_error", "started_at", "completed_at", "config_version"} <= history_columns


def test_ingestion_history_repository_reads_and_updates_records(tmp_path: Path) -> None:
    connection = initialize_metadata_schema(tmp_path / "metadata.db")
    repository = IngestionHistoryRepository(connection)

    first = repository.mark_success(
        source_path="docs/a.pdf",
        source_sha256="sha-1",
        document_id="doc-1",
        config_version="v1",
    )
    second = repository.mark_success(
        source_path="docs/renamed-a.pdf",
        source_sha256="sha-1",
        document_id="doc-1",
        config_version="v1",
    )
    success_candidate = repository.get_success_by_sha256("sha-1")
    failed = repository.mark_failed(
        source_path="docs/renamed-a.pdf",
        source_sha256="sha-1",
        document_id="doc-1",
        error_message="chunk failed",
        config_version="v2",
    )

    assert first["source_sha256"] == "sha-1"
    assert second["source_path"] == "docs/renamed-a.pdf"
    assert repository.get_by_sha256("sha-1")["document_id"] == "doc-1"
    assert success_candidate is not None
    assert success_candidate["status"] == "indexed"
    assert failed["status"] == "failed"
    assert failed["last_error"] == "chunk failed"
    assert repository.get_success_by_sha256("sha-1") is None

    row_count = connection.execute("SELECT COUNT(*) AS count FROM ingestion_history").fetchone()["count"]
    assert row_count == 1


def test_run_sqlite_migrations_is_idempotent(tmp_path: Path) -> None:
    connection = initialize_metadata_schema(tmp_path / "metadata.db")

    assert run_sqlite_migrations(connection) == []

    migration_count = connection.execute(
        "SELECT COUNT(*) AS count FROM schema_migrations"
    ).fetchone()["count"]
    assert migration_count == 5
