"""Collection and document management data access services."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from ragms.runtime.exceptions import RagMSError
from ragms.runtime.settings_models import AppSettings
from ragms.storage.sqlite.connection import create_sqlite_connection
from ragms.storage.sqlite.repositories import DocumentsRepository, ImagesRepository
from ragms.storage.traces import TraceRepository


class DocumentSummaryNotFoundError(RagMSError):
    """Raised when a requested document summary cannot be found."""


class ChunkDetailNotFoundError(RagMSError):
    """Raised when a requested chunk detail cannot be found."""


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
        self.documents_repository = DocumentsRepository(self.connection)
        self.images_repository = ImagesRepository(self.connection)
        self.trace_repository = TraceRepository(settings.dashboard.traces_file)
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

    def get_document_summary(self, document_id: str) -> dict[str, Any]:
        """Return a document summary assembled from registry, index, and image metadata."""

        document = self.documents_repository.get_by_document_id(document_id)
        if document is None:
            raise DocumentSummaryNotFoundError(f"Document not found: {document_id}")

        collection_chunks = self._find_document_chunks(
            document_id=document_id,
            source_path=str(document["source_path"]),
        )
        matched_chunks = [
            chunk
            for chunks in collection_chunks.values()
            for chunk in chunks
        ]
        primary_collection = next(iter(collection_chunks), None)
        image_rows = self.images_repository.list_by_document_id(document_id)

        return {
            "document_id": str(document["document_id"]),
            "source_path": str(document["source_path"]),
            "collections": list(collection_chunks),
            "primary_collection": primary_collection,
            "summary": self._build_document_summary_text(matched_chunks),
            "structure_outline": self._build_structure_outline(matched_chunks),
            "key_metadata": self._extract_key_metadata(document, matched_chunks),
            "ingestion_status": {
                "status": document.get("status"),
                "current_stage": document.get("current_stage"),
                "failure_reason": document.get("failure_reason"),
                "last_ingested_at": document.get("last_ingested_at"),
                "version": document.get("version"),
            },
            "page_summary": self._build_page_summary(matched_chunks, image_rows),
            "image_summary": self._build_image_summary(image_rows),
            "chunk_count": len(matched_chunks),
        }

    def list_documents(
        self,
        *,
        filters: dict[str, Any] | None = None,
        page: int | None = None,
        page_size: int | None = None,
    ) -> dict[str, Any]:
        """Return document summaries with basic filters for the dashboard browser."""

        normalized_filters = dict(filters or {})
        summaries = [self._summarize_document(row) for row in self._list_all_documents()]
        filtered = [summary for summary in summaries if self._matches_document_filters(summary, normalized_filters)]
        filtered.sort(
            key=lambda item: (
                str(item.get("last_ingested_at") or ""),
                str(item.get("updated_at") or ""),
                str(item.get("document_id") or ""),
            ),
            reverse=True,
        )

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
            "documents": paged,
            "pagination": {
                "page": resolved_page,
                "page_size": page_size,
                "total_count": total_count,
                "returned_count": len(paged),
                "has_more": has_more,
            },
            "filters": normalized_filters,
        }

    def get_document_detail(self, document_id: str) -> dict[str, Any]:
        """Return one document detail enriched with chunk and trace navigation."""

        summary = self.get_document_summary(document_id)
        document = self.documents_repository.get_by_document_id(document_id)
        if document is None:
            raise DocumentSummaryNotFoundError(f"Document not found: {document_id}")

        collection_chunks = self._find_document_chunks(
            document_id=document_id,
            source_path=str(document["source_path"]),
        )
        chunks = [
            self._build_chunk_row(chunk, collection_name)
            for collection_name, collection_items in collection_chunks.items()
            for chunk in collection_items
        ]
        related_traces = self._find_related_traces(
            document_id=document_id,
            source_path=str(document["source_path"]),
        )

        return {
            **summary,
            "document": {
                "document_id": str(document["document_id"]),
                "source_path": str(document["source_path"]),
                "source_sha256": document.get("source_sha256"),
                "status": document.get("status"),
                "current_stage": document.get("current_stage"),
                "failure_reason": document.get("failure_reason"),
                "last_ingested_at": document.get("last_ingested_at"),
                "updated_at": document.get("updated_at"),
            },
            "chunks": chunks,
            "related_traces": related_traces,
            "navigation": [
                {"label": "查看 Ingestion Trace", "target_page": "ingestion_trace", "trace_ids": [trace["trace_id"] for trace in related_traces]},
                {"label": "留在数据浏览", "target_page": "data_browser", "document_id": document_id},
            ],
        }

    def get_chunk_detail(self, chunk_id: str, *, collection: str | None = None) -> dict[str, Any]:
        """Return one chunk detail with metadata, references, and previewable images."""

        matches = self._find_chunk_records(chunk_id, collection=collection)
        if not matches:
            raise ChunkDetailNotFoundError(f"Chunk not found: {chunk_id}")

        record = matches[0]
        chunk = record["chunk"]
        metadata = dict(chunk.get("metadata") or {})
        images = self.images_repository.list_by_chunk_ids([chunk_id])
        related_traces = self._find_related_traces(
            document_id=str(chunk.get("document_id") or ""),
            source_path=str(chunk.get("source_path") or ""),
        )
        return {
            "chunk_id": str(chunk.get("chunk_id") or chunk_id),
            "collection": record["collection"],
            "document_id": chunk.get("document_id"),
            "source_path": chunk.get("source_path"),
            "chunk_index": chunk.get("chunk_index"),
            "content": chunk.get("content"),
            "metadata": metadata,
            "page": metadata.get("page"),
            "section_path": list(metadata.get("section_path") or []),
            "tags": list(metadata.get("chunk_tags") or []),
            "references": self._build_chunk_references(chunk, record["collection"]),
            "images": [
                {
                    "image_id": row.get("image_id"),
                    "file_path": row.get("file_path"),
                    "page": row.get("page"),
                    "position": row.get("position"),
                }
                for row in images
            ],
            "related_traces": related_traces,
            "navigation": [
                {"label": "查看文档详情", "target_page": "data_browser", "document_id": chunk.get("document_id")},
                {"label": "查看 Ingestion Trace", "target_page": "ingestion_trace", "trace_ids": [trace["trace_id"] for trace in related_traces]},
            ],
        }

    def get_system_overview_metrics(self) -> dict[str, Any]:
        """Return dashboard-friendly aggregate metrics sourced from local persisted data."""

        collection_statistics = self.get_collection_statistics()
        collections = list(collection_statistics.get("collections") or [])
        document_rows = self._list_all_documents()

        total_documents = len({str(row["document_id"]) for row in document_rows})
        total_chunks = int(collection_statistics.get("chunk_count") or 0)
        total_images = int(collection_statistics.get("image_count") or 0)
        status_counts: dict[str, int] = {}
        for row in document_rows:
            status = str(row.get("status") or "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1

        recent_documents = [
            {
                "document_id": str(row.get("document_id") or ""),
                "source_path": str(row.get("source_path") or ""),
                "status": row.get("status"),
                "current_stage": row.get("current_stage"),
                "last_ingested_at": row.get("last_ingested_at"),
                "updated_at": row.get("updated_at"),
            }
            for row in sorted(
                document_rows,
                key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""),
                reverse=True,
            )[:5]
        ]
        return {
            "collection_count": int(collection_statistics.get("collection_count") or 0),
            "document_count": total_documents,
            "chunk_count": total_chunks,
            "image_count": total_images,
            "status_counts": status_counts,
            "collections": collections,
            "recent_documents": recent_documents,
            "config_summary": {
                "default_collection": self.settings.vector_store.collection,
                "retrieval_strategy": self.settings.retrieval.strategy,
                "rerank_backend": self.settings.retrieval.rerank_backend,
                "trace_log_file": str(self.settings.dashboard.traces_file),
            },
        }

    def get_collection_statistics(self) -> dict[str, Any]:
        """Return collection-level aggregate statistics for dashboard overview pages."""

        collection_payload = self.list_collections()
        collections = list(collection_payload.get("collections") or [])
        return {
            "collection_count": len(collections),
            "document_count": sum(int(item.get("document_count") or 0) for item in collections),
            "chunk_count": sum(int(item.get("chunk_count") or 0) for item in collections),
            "image_count": sum(int(item.get("image_count") or 0) for item in collections),
            "collections": collections,
        }

    def _discover_collection_names(self) -> set[str]:
        names = {str(self.settings.vector_store.collection)}
        if self._bm25_index_dir.is_dir():
            names.update(path.stem for path in self._bm25_index_dir.glob("*.json"))

        if self._image_root.is_dir():
            names.update(path.name for path in self._image_root.iterdir() if path.is_dir())

        try:
            rows = self.connection.execute("SELECT DISTINCT file_path FROM images").fetchall()
        except sqlite3.OperationalError as exc:
            if "no such table: images" in str(exc):
                rows = []
            else:
                raise
        for row in rows:
            file_path = Path(str(row["file_path"])).expanduser().resolve()
            try:
                relative = file_path.relative_to(self._image_root)
            except ValueError:
                continue
            if relative.parts:
                names.add(relative.parts[0])

        return {name for name in names if str(name).strip()}

    def _list_all_documents(self) -> list[dict[str, Any]]:
        try:
            rows = self.connection.execute(
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
                ORDER BY updated_at DESC, created_at DESC
                """
            ).fetchall()
        except sqlite3.OperationalError as exc:
            if "no such table: documents" in str(exc):
                return []
            raise
        return [dict(row) for row in rows]

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

    def _summarize_document(self, document: dict[str, Any]) -> dict[str, Any]:
        collection_chunks = self._find_document_chunks(
            document_id=str(document["document_id"]),
            source_path=str(document["source_path"]),
        )
        chunks = [chunk for values in collection_chunks.values() for chunk in values]
        image_rows = self.images_repository.list_by_document_id(str(document["document_id"]))
        related_traces = self._find_related_traces(
            document_id=str(document["document_id"]),
            source_path=str(document["source_path"]),
        )
        skip_reason = next(
            (
                trace.get("skipped")
                for trace in related_traces
                if trace.get("status") == "skipped" and trace.get("skipped") not in (None, False, "")
            ),
            None,
        )
        page_summary = self._build_page_summary(chunks, image_rows)
        tags = sorted(
            {
                str(tag)
                for chunk in chunks
                for tag in dict(chunk.get("metadata") or {}).get("chunk_tags") or []
                if str(tag).strip()
            }
        )
        return {
            "document_id": str(document["document_id"]),
            "source_path": str(document["source_path"]),
            "collections": list(collection_chunks),
            "primary_collection": next(iter(collection_chunks), None),
            "status": document.get("status"),
            "current_stage": document.get("current_stage"),
            "failure_reason": document.get("failure_reason"),
            "skip_reason": skip_reason,
            "last_ingested_at": document.get("last_ingested_at"),
            "updated_at": document.get("updated_at"),
            "version": document.get("version"),
            "chunk_count": len(chunks),
            "image_count": len(image_rows),
            "pages": page_summary["pages"],
            "page_count": page_summary["page_count"],
            "tags": tags,
            "summary": self._build_document_summary_text(chunks),
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
    ) -> list[dict[str, Any]]:
        rows_by_id = self.documents_repository.list_by_document_ids(document_ids)
        rows_by_source = self.documents_repository.list_by_source_paths(source_paths)
        merged: dict[str, dict[str, Any]] = {}
        for row in rows_by_id + rows_by_source:
            merged[str(row["document_id"])] = row
        return list(merged.values())

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

    def _find_document_chunks(
        self,
        *,
        document_id: str,
        source_path: str,
    ) -> dict[str, list[dict[str, Any]]]:
        matches: dict[str, list[dict[str, Any]]] = {}
        normalized_source = str(source_path).strip()
        for collection_name in sorted(self._discover_collection_names()):
            snapshot = self._load_bm25_snapshot(collection_name)
            chunk_matches = []
            for payload in (snapshot.get("documents") or {}).values():
                payload_document_id = str(payload.get("document_id") or "").strip()
                payload_source_path = str(payload.get("source_path") or "").strip()
                if payload_document_id == document_id or payload_source_path == normalized_source:
                    chunk_matches.append(dict(payload))
            if chunk_matches:
                chunk_matches.sort(key=lambda item: int(item.get("chunk_index", 0) or 0))
                matches[collection_name] = chunk_matches
        return matches

    def _find_chunk_records(
        self,
        chunk_id: str,
        *,
        collection: str | None = None,
    ) -> list[dict[str, Any]]:
        collection_names = [collection] if collection is not None else sorted(self._discover_collection_names())
        matches: list[dict[str, Any]] = []
        for collection_name in collection_names:
            snapshot = self._load_bm25_snapshot(collection_name)
            payload = dict((snapshot.get("documents") or {}).get(chunk_id) or {})
            if payload:
                matches.append({"collection": collection_name, "chunk": payload})
        return matches

    def _find_related_traces(self, *, document_id: str, source_path: str) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        for trace in self.trace_repository.list_traces(limit=100):
            if str(trace.get("trace_type") or "") != "ingestion":
                continue
            if document_id and str(trace.get("document_id") or "") == document_id:
                matches.append(self._summarize_related_trace(trace))
                continue
            if source_path and str(trace.get("source_path") or "") == source_path:
                matches.append(self._summarize_related_trace(trace))
        return matches

    @staticmethod
    def _summarize_related_trace(trace: dict[str, Any]) -> dict[str, Any]:
        return {
            "trace_id": trace.get("trace_id"),
            "trace_type": trace.get("trace_type"),
            "status": trace.get("status"),
            "started_at": trace.get("started_at"),
            "finished_at": trace.get("finished_at"),
            "duration_ms": trace.get("duration_ms"),
            "skipped": trace.get("skipped"),
            "target_page": "ingestion_trace",
        }

    def _build_chunk_row(self, chunk: dict[str, Any], collection_name: str) -> dict[str, Any]:
        metadata = dict(chunk.get("metadata") or {})
        return {
            "chunk_id": str(chunk.get("chunk_id") or ""),
            "collection": collection_name,
            "chunk_index": chunk.get("chunk_index"),
            "page": metadata.get("page"),
            "section_path": list(metadata.get("section_path") or []),
            "tags": list(metadata.get("chunk_tags") or []),
            "summary": str(metadata.get("chunk_summary") or "").strip() or self._truncate(str(chunk.get("content") or "")),
            "target_page": "data_browser",
        }

    @staticmethod
    def _build_chunk_references(chunk: dict[str, Any], collection_name: str) -> dict[str, Any]:
        metadata = dict(chunk.get("metadata") or {})
        return {
            "collection": collection_name,
            "source_path": chunk.get("source_path"),
            "page": metadata.get("page"),
            "section_path": list(metadata.get("section_path") or []),
            "chunk_index": chunk.get("chunk_index"),
        }

    @staticmethod
    def _build_document_summary_text(chunks: list[dict[str, Any]]) -> str | None:
        if not chunks:
            return None
        for chunk in chunks:
            metadata = dict(chunk.get("metadata") or {})
            summary = str(metadata.get("chunk_summary") or "").strip()
            if summary:
                return summary
        content = str(chunks[0].get("content") or "").strip()
        if not content:
            return None
        return content[:240]

    @staticmethod
    def _build_structure_outline(chunks: list[dict[str, Any]]) -> list[str]:
        outline: list[str] = []
        for chunk in chunks:
            metadata = dict(chunk.get("metadata") or {})
            section_path = [str(item).strip() for item in metadata.get("section_path") or [] if str(item).strip()]
            if section_path:
                label = " / ".join(section_path)
            else:
                label = str(metadata.get("chunk_title") or metadata.get("title") or "").strip()
            if label and label not in outline:
                outline.append(label)
        return outline

    @staticmethod
    def _extract_key_metadata(document: dict[str, Any], chunks: list[dict[str, Any]]) -> dict[str, Any]:
        first_metadata = dict(chunks[0].get("metadata") or {}) if chunks else {}
        result = {
            "source_sha256": document.get("source_sha256"),
            "doc_type": first_metadata.get("doc_type"),
            "title": first_metadata.get("title"),
            "chunk_title": first_metadata.get("chunk_title"),
            "chunk_tags": list(first_metadata.get("chunk_tags") or []),
        }
        return {key: value for key, value in result.items() if value not in (None, [], "")}

    @staticmethod
    def _build_page_summary(chunks: list[dict[str, Any]], image_rows: list[dict[str, Any]]) -> dict[str, Any]:
        pages = {
            int(page)
            for page in [
                *(dict(chunk.get("metadata") or {}).get("page") for chunk in chunks),
                *(row.get("page") for row in image_rows),
            ]
            if page is not None
        }
        return {
            "pages": sorted(pages),
            "page_count": len(pages),
        }

    @staticmethod
    def _build_image_summary(image_rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "image_count": len(image_rows),
            "images": [
                {
                    "image_id": row.get("image_id"),
                    "chunk_id": row.get("chunk_id"),
                    "file_path": row.get("file_path"),
                    "source_path": row.get("source_path"),
                    "page": row.get("page"),
                    "position": row.get("position"),
                }
                for row in image_rows
            ],
        }

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

    def _matches_document_filters(self, summary: dict[str, Any], filters: dict[str, Any]) -> bool:
        if not filters:
            return True

        collection = str(filters.get("collection") or "").strip()
        if collection and collection not in (summary.get("collections") or []):
            return False

        status = str(filters.get("status") or "").strip()
        if status and str(summary.get("status") or "") != status:
            return False

        keyword = str(filters.get("keyword") or "").strip().lower()
        if keyword:
            haystacks = [
                str(summary.get("document_id") or ""),
                str(summary.get("source_path") or ""),
                str(summary.get("summary") or ""),
            ]
            if not any(keyword in item.lower() for item in haystacks):
                return False

        page = filters.get("page")
        if page not in (None, ""):
            try:
                resolved_page = int(page)
            except (TypeError, ValueError):
                resolved_page = None
            if resolved_page is not None and resolved_page not in (summary.get("pages") or []):
                return False

        tag = str(filters.get("tag") or "").strip()
        if tag and tag not in (summary.get("tags") or []):
            return False

        return True

    @staticmethod
    def _truncate(value: str, limit: int = 120) -> str:
        text = value.strip()
        if len(text) <= limit:
            return text
        return f"{text[:limit].rstrip()}..."
