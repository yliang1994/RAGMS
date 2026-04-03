"""Repository for persisted image-to-document/chunk mappings."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from typing import Any


class ImagesRepository:
    """Store image file mappings for dashboard, delete, and rebuild workflows."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self._ensure_table()

    def get(self, *, image_id: str, chunk_id: str) -> dict[str, Any] | None:
        """Return one image mapping row by image id and chunk id."""

        row = self.connection.execute(
            """
            SELECT
                image_id,
                document_id,
                chunk_id,
                file_path,
                source_path,
                image_hash,
                page,
                position_json,
                created_at,
                updated_at
            FROM images
            WHERE image_id = ? AND chunk_id = ?
            """,
            (image_id, chunk_id),
        ).fetchone()
        if row is None:
            return None
        payload = dict(row)
        payload["position"] = json.loads(str(payload.pop("position_json") or "{}"))
        return payload

    def list_by_document_id(self, document_id: str) -> list[dict[str, Any]]:
        """Return all persisted image mappings for a document."""

        rows = self.connection.execute(
            """
            SELECT
                image_id,
                document_id,
                chunk_id,
                file_path,
                source_path,
                image_hash,
                page,
                position_json,
                created_at,
                updated_at
            FROM images
            WHERE document_id = ?
            ORDER BY image_id, chunk_id
            """,
            (document_id,),
        ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            payload = dict(row)
            payload["position"] = json.loads(str(payload.pop("position_json") or "{}"))
            results.append(payload)
        return results

    def upsert_image(
        self,
        *,
        image_id: str,
        document_id: str,
        chunk_id: str,
        file_path: str,
        source_path: str,
        image_hash: str,
        page: int | None = None,
        position: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Insert or update one persisted image mapping."""

        timestamp = _utc_now()
        position_json = json.dumps(position or {}, ensure_ascii=False, sort_keys=True)
        existing = self.get(image_id=image_id, chunk_id=chunk_id)
        if existing is None:
            self.connection.execute(
                """
                INSERT INTO images (
                    image_id,
                    document_id,
                    chunk_id,
                    file_path,
                    source_path,
                    image_hash,
                    page,
                    position_json,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    image_id,
                    document_id,
                    chunk_id,
                    file_path,
                    source_path,
                    image_hash,
                    page,
                    position_json,
                    timestamp,
                    timestamp,
                ),
            )
        else:
            self.connection.execute(
                """
                UPDATE images
                SET
                    document_id = ?,
                    file_path = ?,
                    source_path = ?,
                    image_hash = ?,
                    page = ?,
                    position_json = ?,
                    updated_at = ?
                WHERE image_id = ? AND chunk_id = ?
                """,
                (
                    document_id,
                    file_path,
                    source_path,
                    image_hash,
                    page,
                    position_json,
                    timestamp,
                    image_id,
                    chunk_id,
                ),
            )
        self.connection.commit()
        stored = self.get(image_id=image_id, chunk_id=chunk_id)
        if stored is None:  # pragma: no cover - defensive boundary
            raise RuntimeError("Failed to persist image metadata record")
        return stored

    def _ensure_table(self) -> None:
        """Create the images table if it does not exist."""

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                source_path TEXT NOT NULL,
                image_hash TEXT NOT NULL,
                page INTEGER,
                position_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(image_id, chunk_id)
            )
            """
        )
        self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_images_document_id
            ON images(document_id)
            """
        )
        self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_images_image_id
            ON images(image_id)
            """
        )
        self.connection.commit()


def _utc_now() -> str:
    """Return an ISO8601 timestamp in UTC."""

    return datetime.now(UTC).isoformat()
