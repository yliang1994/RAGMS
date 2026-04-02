"""Repository for persisted document registration records."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import Any


class DocumentsRepository:
    """Store and query document-level ingestion metadata in SQLite."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def get_by_document_id(self, document_id: str) -> dict[str, Any] | None:
        """Return a document row by document id."""

        row = self.connection.execute(
            """
            SELECT document_id, source_path, source_sha256, status, current_stage, created_at, updated_at
            FROM documents
            WHERE document_id = ?
            """,
            (document_id,),
        ).fetchone()
        return dict(row) if row is not None else None

    def get_by_source_path(self, source_path: str) -> dict[str, Any] | None:
        """Return the latest document row by source path."""

        row = self.connection.execute(
            """
            SELECT document_id, source_path, source_sha256, status, current_stage, created_at, updated_at
            FROM documents
            WHERE source_path = ?
            ORDER BY updated_at DESC, created_at DESC
            LIMIT 1
            """,
            (source_path,),
        ).fetchone()
        return dict(row) if row is not None else None

    def upsert_document(
        self,
        *,
        document_id: str,
        source_path: str,
        source_sha256: str,
        status: str,
        current_stage: str,
    ) -> dict[str, Any]:
        """Insert or update a document registration row."""

        timestamp = _utc_now()
        existing = self.get_by_document_id(document_id)
        if existing is None:
            self.connection.execute(
                """
                INSERT INTO documents (
                    document_id,
                    source_path,
                    source_sha256,
                    status,
                    current_stage,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (document_id, source_path, source_sha256, status, current_stage, timestamp, timestamp),
            )
        else:
            self.connection.execute(
                """
                UPDATE documents
                SET source_path = ?, source_sha256 = ?, status = ?, current_stage = ?, updated_at = ?
                WHERE document_id = ?
                """,
                (source_path, source_sha256, status, current_stage, timestamp, document_id),
            )
        self.connection.commit()
        stored = self.get_by_document_id(document_id)
        if stored is None:  # pragma: no cover - defensive boundary
            raise RuntimeError("Failed to persist document metadata record")
        return stored

    def update_status(
        self,
        document_id: str,
        *,
        status: str,
        current_stage: str | None = None,
    ) -> dict[str, Any]:
        """Update document status and optionally its current stage."""

        existing = self.get_by_document_id(document_id)
        if existing is None:
            raise KeyError(f"Unknown document: {document_id}")

        timestamp = _utc_now()
        resolved_stage = current_stage if current_stage is not None else str(existing["current_stage"])
        self.connection.execute(
            """
            UPDATE documents
            SET status = ?, current_stage = ?, updated_at = ?
            WHERE document_id = ?
            """,
            (status, resolved_stage, timestamp, document_id),
        )
        self.connection.commit()
        stored = self.get_by_document_id(document_id)
        if stored is None:  # pragma: no cover - defensive boundary
            raise RuntimeError("Failed to update document metadata record")
        return stored


def _utc_now() -> str:
    """Return an ISO8601 timestamp in UTC."""

    return datetime.now(UTC).isoformat()
