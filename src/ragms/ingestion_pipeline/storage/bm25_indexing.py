"""Storage writer for persistent BM25 sparse indexes."""

from __future__ import annotations

from typing import Any

from ragms.storage.indexes import BM25Indexer


class BM25StorageWriter:
    """Persist sparse chunk records into the BM25 index."""

    def __init__(self, indexer: BM25Indexer) -> None:
        self.indexer = indexer

    def index(self, records: list[Any]) -> dict[str, Any]:
        """Index chunk records and return a stable persistence summary."""

        indexed_ids: list[str] = []
        for record in records:
            entry = self.indexer.index_document(record)
            indexed_ids.append(str(entry["chunk_id"]))
        snapshot = self.indexer.snapshot()
        return {
            "indexed_ids": indexed_ids,
            "indexed_count": len(indexed_ids),
            "document_count": int(snapshot["document_count"]),
            "average_document_length": float(snapshot["average_document_length"]),
            "index_path": str(self.indexer.index_path),
        }
