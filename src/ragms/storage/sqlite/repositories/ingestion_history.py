"""Repository for ingestion history deduplication records."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import Any


class IngestionHistoryRepository:
    """Store and query content-hash based ingestion history records."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def get_by_sha256(self, source_sha256: str) -> dict[str, Any] | None:
        """Return a single ingestion-history row for a content hash."""

        row = self.connection.execute(
            """
            SELECT id, source_path, source_sha256, document_id, status, created_at, updated_at
            FROM ingestion_history
            WHERE source_sha256 = ?
            """,
            (source_sha256,),
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

        timestamp = _utc_now()
        existing = self.get_by_sha256(source_sha256)
        if existing is None:
            self.connection.execute(
                """
                INSERT INTO ingestion_history (
                    source_path,
                    source_sha256,
                    document_id,
                    status,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (source_path, source_sha256, document_id, status, timestamp, timestamp),
            )
        else:
            self.connection.execute(
                """
                UPDATE ingestion_history
                SET source_path = ?, document_id = ?, status = ?, updated_at = ?
                WHERE source_sha256 = ?
                """,
                (source_path, document_id, status, timestamp, source_sha256),
            )
        self.connection.commit()
        stored = self.get_by_sha256(source_sha256)
        if stored is None:  # pragma: no cover - defensive boundary
            raise RuntimeError("Failed to persist ingestion history record")
        return stored


def _utc_now() -> str:
    """Return an ISO8601 timestamp in UTC."""

    return datetime.now(UTC).isoformat()
