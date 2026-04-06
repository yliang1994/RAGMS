"""Collection and document management data access services."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from ragms.runtime.settings_models import AppSettings
from ragms.storage.sqlite.connection import create_sqlite_connection


class DataService:
    """Read management-oriented summaries from persisted local metadata."""

    def __init__(
        self,
        settings: AppSettings,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        self.settings = settings
        self.connection = connection or create_sqlite_connection(settings.storage.sqlite.path)
        self._bm25_index_dir = settings.paths.data_dir / "indexes" / "sparse"
        self._image_root = (settings.paths.data_dir / "images").expanduser().resolve()

    def list_collections(
        self,
        *,
        filters: dict[str, Any] | None = None,
        page: int | None = None,
        page_size: int | None = None,
    ) -> dict[str, Any]:
        """Return collection summaries with optional compatibility filtering and paging."""

        normalized_filters = dict(filters or {})
        summaries = [self._summarize_collection(name) for name in self._discover_collection_names()]
        filtered = [summary for summary in summaries if self._matches_filters(summary, normalized_filters)]
        filtered.sort(key=lambda item: item["name"])

        total_count = len(filtered)
        if page_size is not None:
            resolved_page = page or 1
            start = (resolved_page - 1) * page_size
            end = start + page_size
            paged = filtered[start:end]
            has_more = end < total_count
        else:
            resolved_page = page
            paged = filtered
            has_more = False

        return {
            "collections": paged,
            "pagination": {
                "page": resolved_page,
                "page_size": page_size,
                "total_count": total_count,
                "returned_count": len(paged),
                "has_more": has_more,
            },
            "filters": normalized_filters,
        }

    def _discover_collection_names(self) -> set[str]:
        names = {str(self.settings.vector_store.collection)}
        if self._bm25_index_dir.is_dir():
            names.update(path.stem for path in self._bm25_index_dir.glob("*.json"))

        if self._image_root.is_dir():
            names.update(path.name for path in self._image_root.iterdir() if path.is_dir())

        rows = self.connection.execute("SELECT DISTINCT file_path FROM images").fetchall()
        for row in rows:
            file_path = Path(str(row["file_path"])).expanduser().resolve()
            try:
                relative = file_path.relative_to(self._image_root)
            except ValueError:
                continue
            if relative.parts:
                names.add(relative.parts[0])

        return {name for name in names if str(name).strip()}

    def _summarize_collection(self, collection_name: str) -> dict[str, Any]:
        snapshot = self._load_bm25_snapshot(collection_name)
        documents = list((snapshot.get("documents") or {}).values())
        document_ids = sorted(
            {
                str(item.get("document_id") or "").strip()
                for item in documents
                if str(item.get("document_id") or "").strip()
            }
        )
        source_paths = sorted(
            {
                str(item.get("source_path") or "").strip()
                for item in documents
                if str(item.get("source_path") or "").strip()
            }
        )
        document_rows = self._fetch_document_rows(document_ids=document_ids, source_paths=source_paths)
        latest_updated_at = max(
            (
                str(row["last_ingested_at"] or row["updated_at"] or "")
                for row in document_rows
                if str(row["last_ingested_at"] or row["updated_at"] or "").strip()
            ),
            default=None,
        )
        image_count = self._count_images_for_collection(collection_name)

        return {
            "name": collection_name,
            "document_count": len({str(row["document_id"]) for row in document_rows}) or len(document_ids),
            "chunk_count": len(documents),
            "image_count": image_count,
            "latest_updated_at": latest_updated_at,
        }

    def _load_bm25_snapshot(self, collection_name: str) -> dict[str, Any]:
        index_path = self._bm25_index_dir / f"{collection_name}.json"
        if not index_path.is_file():
            return {"collection": collection_name, "documents": {}}
        return json.loads(index_path.read_text(encoding="utf-8"))

    def _fetch_document_rows(
        self,
        *,
        document_ids: list[str],
        source_paths: list[str],
    ) -> list[sqlite3.Row]:
        conditions: list[str] = []
        parameters: list[str] = []
        if document_ids:
            conditions.append(f"document_id IN ({', '.join('?' for _ in document_ids)})")
            parameters.extend(document_ids)
        if source_paths:
            conditions.append(f"source_path IN ({', '.join('?' for _ in source_paths)})")
            parameters.extend(source_paths)
        if not conditions:
            return []

        query = """
            SELECT document_id, source_path, last_ingested_at, updated_at
            FROM documents
            WHERE {conditions}
        """.format(conditions=" OR ".join(conditions))
        return list(self.connection.execute(query, tuple(parameters)).fetchall())

    def _count_images_for_collection(self, collection_name: str) -> int:
        collection_root = str((self._image_root / collection_name).resolve())
        row = self.connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM images
            WHERE file_path = ? OR file_path LIKE ?
            """,
            (collection_root, f"{collection_root}/%"),
        ).fetchone()
        return int(row["count"]) if row is not None else 0

    @staticmethod
    def _matches_filters(summary: dict[str, Any], filters: dict[str, Any]) -> bool:
        if not filters:
            return True

        aliases = {
            "collection": "name",
            "collection_name": "name",
        }
        for raw_key, expected in filters.items():
            key = aliases.get(str(raw_key), str(raw_key))
            actual = summary.get(key)
            if isinstance(expected, list):
                if actual not in expected:
                    return False
                continue
            if actual != expected:
                return False
        return True
