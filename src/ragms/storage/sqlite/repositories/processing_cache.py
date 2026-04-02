"""Repository for reusable processing artifacts such as image captions."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import Any


class ProcessingCacheRepository:
    """Store processing-cache entries keyed by content hash and config version."""

    CACHE_TYPE_IMAGE_CAPTION = "image_caption"

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self._ensure_table()

    def get_caption(
        self,
        *,
        image_hash: str,
        model: str,
        prompt_version: str,
    ) -> dict[str, Any] | None:
        """Return a cached caption entry for the given image/config tuple."""

        row = self.connection.execute(
            """
            SELECT
                id,
                cache_type,
                image_hash,
                model,
                prompt_version,
                image_path,
                payload,
                created_at,
                updated_at
            FROM processing_cache
            WHERE cache_type = ?
              AND image_hash = ?
              AND model = ?
              AND prompt_version = ?
            """,
            (
                self.CACHE_TYPE_IMAGE_CAPTION,
                image_hash,
                model,
                prompt_version,
            ),
        ).fetchone()
        return dict(row) if row is not None else None

    def upsert_caption(
        self,
        *,
        image_hash: str,
        model: str,
        prompt_version: str,
        image_path: str,
        caption: str,
    ) -> dict[str, Any]:
        """Insert or update a caption cache entry and return the stored row."""

        existing = self.get_caption(
            image_hash=image_hash,
            model=model,
            prompt_version=prompt_version,
        )
        timestamp = _utc_now()
        if existing is None:
            self.connection.execute(
                """
                INSERT INTO processing_cache (
                    cache_type,
                    image_hash,
                    model,
                    prompt_version,
                    image_path,
                    payload,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.CACHE_TYPE_IMAGE_CAPTION,
                    image_hash,
                    model,
                    prompt_version,
                    image_path,
                    caption,
                    timestamp,
                    timestamp,
                ),
            )
        else:
            self.connection.execute(
                """
                UPDATE processing_cache
                SET
                    image_path = ?,
                    payload = ?,
                    updated_at = ?
                WHERE cache_type = ?
                  AND image_hash = ?
                  AND model = ?
                  AND prompt_version = ?
                """,
                (
                    image_path,
                    caption,
                    timestamp,
                    self.CACHE_TYPE_IMAGE_CAPTION,
                    image_hash,
                    model,
                    prompt_version,
                ),
            )
        self.connection.commit()
        stored = self.get_caption(
            image_hash=image_hash,
            model=model,
            prompt_version=prompt_version,
        )
        if stored is None:  # pragma: no cover - defensive boundary
            raise RuntimeError("Failed to persist processing cache entry")
        return stored

    def _ensure_table(self) -> None:
        """Create the processing cache table if it does not exist."""

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS processing_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cache_type TEXT NOT NULL,
                image_hash TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                image_path TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(cache_type, image_hash, model, prompt_version)
            )
            """
        )
        self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_processing_cache_lookup
            ON processing_cache(cache_type, image_hash, model, prompt_version)
            """
        )
        self.connection.commit()


def _utc_now() -> str:
    """Return an ISO8601 timestamp in UTC."""

    return datetime.now(UTC).isoformat()
