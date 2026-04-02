"""Repository for ingestion history deduplication records."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import Any


class IngestionHistoryRepository:
    """Store and query content-hash based ingestion history records."""

    SUCCESS_STATUSES = {"indexed", "skipped", "success"}

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def get_by_sha256(self, source_sha256: str) -> dict[str, Any] | None:
        """Return a single ingestion-history row for a content hash."""

        row = self.connection.execute(
            """
            SELECT
                id,
                source_path,
                source_sha256,
                document_id,
                status,
                last_error,
                started_at,
                completed_at,
                config_version,
                created_at,
                updated_at
            FROM ingestion_history
            WHERE source_sha256 = ?
            """,
            (source_sha256,),
        ).fetchone()
        return dict(row) if row is not None else None

    def get_success_by_sha256(self, source_sha256: str) -> dict[str, Any] | None:
        """Return a successful ingestion-history row for a content hash."""

        placeholders = ", ".join("?" for _ in self.SUCCESS_STATUSES)
        row = self.connection.execute(
            f"""
            SELECT
                id,
                source_path,
                source_sha256,
                document_id,
                status,
                last_error,
                started_at,
                completed_at,
                config_version,
                created_at,
                updated_at
            FROM ingestion_history
            WHERE source_sha256 = ?
              AND status IN ({placeholders})
            """,
            (source_sha256, *sorted(self.SUCCESS_STATUSES)),
        ).fetchone()
        return dict(row) if row is not None else None

    def record_processed(
        self,
        *,
        source_path: str,
        source_sha256: str,
        status: str = "indexed",
        document_id: str | None = None,
    ) -> dict[str, Any]:
        """Insert or update an ingestion-history record and return the stored row."""

        return self.mark_success(
            source_path=source_path,
            source_sha256=source_sha256,
            status=status,
            document_id=document_id,
        )

    def mark_success(
        self,
        *,
        source_path: str,
        source_sha256: str,
        status: str = "indexed",
        document_id: str | None = None,
        config_version: str | None = None,
        started_at: str | None = None,
        completed_at: str | None = None,
    ) -> dict[str, Any]:
        """Persist a successful ingestion outcome for a content hash."""

        if status not in self.SUCCESS_STATUSES:
            raise ValueError(f"Unsupported success status: {status}")
        timestamp = _utc_now()
        return self._upsert_record(
            source_path=source_path,
            source_sha256=source_sha256,
            document_id=document_id,
            status=status,
            last_error=None,
            started_at=started_at or timestamp,
            completed_at=completed_at or timestamp,
            config_version=config_version,
            updated_at=timestamp,
        )

    def mark_failed(
        self,
        *,
        source_path: str,
        source_sha256: str,
        error_message: str,
        document_id: str | None = None,
        config_version: str | None = None,
        started_at: str | None = None,
        completed_at: str | None = None,
    ) -> dict[str, Any]:
        """Persist a failed ingestion outcome for a content hash."""

        timestamp = _utc_now()
        return self._upsert_record(
            source_path=source_path,
            source_sha256=source_sha256,
            document_id=document_id,
            status="failed",
            last_error=error_message,
            started_at=started_at or timestamp,
            completed_at=completed_at,
            config_version=config_version,
            updated_at=timestamp,
        )

    def delete_by_document_id(self, document_id: str) -> int:
        """Delete ingestion-history rows belonging to a document id."""

        result = self.connection.execute(
            "DELETE FROM ingestion_history WHERE document_id = ?",
            (document_id,),
        )
        self.connection.commit()
        return int(result.rowcount)

    def _upsert_record(
        self,
        *,
        source_path: str,
        source_sha256: str,
        document_id: str | None,
        status: str,
        last_error: str | None,
        started_at: str | None,
        completed_at: str | None,
        config_version: str | None,
        updated_at: str,
    ) -> dict[str, Any]:
        """Insert or update an ingestion-history record and return the stored row."""

        existing = self.get_by_sha256(source_sha256)
        created_at = updated_at
        if existing is None:
            self.connection.execute(
                """
                INSERT INTO ingestion_history (
                    source_path,
                    source_sha256,
                    document_id,
                    status,
                    last_error,
                    started_at,
                    completed_at,
                    config_version,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_path,
                    source_sha256,
                    document_id,
                    status,
                    last_error,
                    started_at,
                    completed_at,
                    config_version,
                    created_at,
                    updated_at,
                ),
            )
        else:
            existing_started_at = existing.get("started_at")
            self.connection.execute(
                """
                UPDATE ingestion_history
                SET
                    source_path = ?,
                    document_id = ?,
                    status = ?,
                    last_error = ?,
                    started_at = ?,
                    completed_at = ?,
                    config_version = ?,
                    updated_at = ?
                WHERE source_sha256 = ?
                """,
                (
                    source_path,
                    document_id,
                    status,
                    last_error,
                    started_at or existing_started_at,
                    completed_at,
                    config_version,
                    updated_at,
                    source_sha256,
                ),
            )

        self.connection.commit()
        stored = self.get_by_sha256(source_sha256)
        if stored is None:  # pragma: no cover - defensive boundary
            raise RuntimeError("Failed to persist ingestion history record")
        return stored


def _utc_now() -> str:
    """Return an ISO8601 timestamp in UTC."""

    return datetime.now(UTC).isoformat()
