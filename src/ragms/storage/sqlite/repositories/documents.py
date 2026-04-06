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

        row = self._safe_fetchone(
            """
            SELECT
                document_id,
                source_path,
                source_sha256,
                status,
                current_stage,
                failure_reason,
                last_ingested_at,
                version,
                created_at,
                updated_at
            FROM documents
            WHERE document_id = ?
            """,
            (document_id,),
        )
        return dict(row) if row is not None else None

    def get_by_source_path(self, source_path: str) -> dict[str, Any] | None:
        """Return the latest document row by source path."""

        row = self._safe_fetchone(
            """
            SELECT
                document_id,
                source_path,
                source_sha256,
                status,
                current_stage,
                failure_reason,
                last_ingested_at,
                version,
                created_at,
                updated_at
            FROM documents
            WHERE source_path = ?
            ORDER BY updated_at DESC, created_at DESC
            LIMIT 1
            """,
            (source_path,),
        )
        return dict(row) if row is not None else None

    def list_by_document_ids(self, document_ids: list[str]) -> list[dict[str, Any]]:
        """Return document rows for a set of document ids."""

        return self._list_where("document_id", document_ids)

    def list_by_source_paths(self, source_paths: list[str]) -> list[dict[str, Any]]:
        """Return document rows for a set of source paths."""

        return self._list_where("source_path", source_paths)

    def upsert_document(
        self,
        *,
        document_id: str,
        source_path: str,
        source_sha256: str,
        status: str,
        current_stage: str,
        version: int | None = None,
        failure_reason: str | None = None,
        last_ingested_at: str | None = None,
    ) -> dict[str, Any]:
        """Insert or update a document registration row."""

        timestamp = _utc_now()
        existing = self.get_by_document_id(document_id)
        if existing is None:
            resolved_version = int(version or 1)
            self.connection.execute(
                """
                INSERT INTO documents (
                    document_id,
                    source_path,
                    source_sha256,
                    status,
                    current_stage,
                    failure_reason,
                    last_ingested_at,
                    version,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    source_path,
                    source_sha256,
                    status,
                    current_stage,
                    failure_reason,
                    last_ingested_at,
                    resolved_version,
                    timestamp,
                    timestamp,
                ),
            )
        else:
            existing_version = int(existing.get("version") or 1)
            resolved_version = int(version or existing_version)
            if source_sha256 != str(existing["source_sha256"]) and version is None:
                resolved_version = existing_version + 1
            self.connection.execute(
                """
                UPDATE documents
                SET
                    source_path = ?,
                    source_sha256 = ?,
                    status = ?,
                    current_stage = ?,
                    failure_reason = ?,
                    last_ingested_at = ?,
                    version = ?,
                    updated_at = ?
                WHERE document_id = ?
                """,
                (
                    source_path,
                    source_sha256,
                    status,
                    current_stage,
                    failure_reason,
                    last_ingested_at,
                    resolved_version,
                    timestamp,
                    document_id,
                ),
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
        failure_reason: str | None = None,
        last_ingested_at: str | None = None,
    ) -> dict[str, Any]:
        """Update document status and optionally its current stage."""

        existing = self.get_by_document_id(document_id)
        if existing is None:
            raise KeyError(f"Unknown document: {document_id}")

        timestamp = _utc_now()
        resolved_stage = current_stage if current_stage is not None else str(existing["current_stage"])
        resolved_failure_reason = failure_reason if status == "failed" else None
        resolved_last_ingested_at = last_ingested_at
        if resolved_last_ingested_at is None and status in {"indexed", "skipped"}:
            resolved_last_ingested_at = timestamp
        self.connection.execute(
            """
            UPDATE documents
            SET
                status = ?,
                current_stage = ?,
                failure_reason = ?,
                last_ingested_at = COALESCE(?, last_ingested_at),
                updated_at = ?
            WHERE document_id = ?
            """,
            (
                status,
                resolved_stage,
                resolved_failure_reason,
                resolved_last_ingested_at,
                timestamp,
                document_id,
            ),
        )
        self.connection.commit()
        stored = self.get_by_document_id(document_id)
        if stored is None:  # pragma: no cover - defensive boundary
            raise RuntimeError("Failed to update document metadata record")
        return stored

    def _list_where(self, field: str, values: list[str]) -> list[dict[str, Any]]:
        """Return document rows matching one field against multiple values."""

        normalized = [str(value).strip() for value in values if str(value).strip()]
        if not normalized:
            return []

        placeholders = ", ".join("?" for _ in normalized)
        rows = self._safe_fetchall(
            f"""
            SELECT
                document_id,
                source_path,
                source_sha256,
                status,
                current_stage,
                failure_reason,
                last_ingested_at,
                version,
                created_at,
                updated_at
            FROM documents
            WHERE {field} IN ({placeholders})
            ORDER BY updated_at DESC, created_at DESC
            """,
            tuple(normalized),
        )
        return [dict(row) for row in rows]

    def _safe_fetchone(self, query: str, parameters: tuple[Any, ...]) -> sqlite3.Row | None:
        """Fetch one row and treat missing tables as empty results."""

        try:
            return self.connection.execute(query, parameters).fetchone()
        except sqlite3.OperationalError as exc:
            if "no such table: documents" in str(exc):
                return None
            raise

    def _safe_fetchall(self, query: str, parameters: tuple[Any, ...]) -> list[sqlite3.Row]:
        """Fetch all rows and treat missing tables as empty results."""

        try:
            return list(self.connection.execute(query, parameters).fetchall())
        except sqlite3.OperationalError as exc:
            if "no such table: documents" in str(exc):
                return []
            raise


def _utc_now() -> str:
    """Return an ISO8601 timestamp in UTC."""

    return datetime.now(UTC).isoformat()
