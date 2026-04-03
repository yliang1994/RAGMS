"""Stable vector-store upsert for chunk records."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from ragms.libs.abstractions import BaseVectorStore
from ragms.runtime.exceptions import RagMSError


class VectorUpsertError(RagMSError):
    """Raised when vector-store persistence cannot complete safely."""


class VectorUpsert:
    """Write chunk records to the vector store with batching and idempotence guards."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        *,
        batch_size: int = 32,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        self.vector_store = vector_store
        self.batch_size = batch_size
        self._written_fingerprints: dict[str, str] = {}

    def write(self, records: list[Any]) -> list[str]:
        """Upsert chunk records in stable batches and roll back on later failure."""

        if not records:
            return []

        prepared_records = self._deduplicate(records)
        written_ids: list[str] = []
        try:
            for batch in self._iter_batches(prepared_records):
                ids = [str(record.chunk_id) for record in batch]
                vectors = [list(record.dense_vector) for record in batch]
                documents = [str(record.content) for record in batch]
                metadatas = [self._build_vector_metadata(record) for record in batch]
                self.vector_store.add(
                    ids,
                    vectors,
                    documents=documents,
                    metadatas=metadatas,
                )
                written_ids.extend(ids)
                for record in batch:
                    self._written_fingerprints[str(record.chunk_id)] = str(record.content_hash)
        except Exception as exc:
            self._rollback(written_ids)
            raise VectorUpsertError("Vector upsert failed") from exc
        return written_ids

    def _deduplicate(self, records: list[Any]) -> list[Any]:
        """Collapse already-written records while preserving first-seen order."""

        deduplicated: list[Any] = []
        seen_in_call: dict[str, str] = {}
        for record in records:
            chunk_id = str(record.chunk_id)
            content_hash = str(record.content_hash)
            previous_hash = seen_in_call.get(chunk_id)
            if previous_hash is not None and previous_hash != content_hash:
                raise VectorUpsertError(
                    f"Conflicting records share chunk_id {chunk_id} but differ in content_hash"
                )
            seen_in_call[chunk_id] = content_hash

            if self._written_fingerprints.get(chunk_id) == content_hash:
                continue
            if previous_hash is None:
                deduplicated.append(record)
        return deduplicated

    def _iter_batches(self, records: list[Any]) -> list[list[Any]]:
        """Split records into stable write batches."""

        return [
            records[index : index + self.batch_size]
            for index in range(0, len(records), self.batch_size)
        ]

    def _rollback(self, ids: list[str]) -> None:
        """Best-effort rollback for partially written batches."""

        if not ids:
            return
        try:
            self.vector_store.delete(ids)
        except Exception:
            return
        for item_id in ids:
            self._written_fingerprints.pop(item_id, None)

    @staticmethod
    def _build_vector_metadata(record: Any) -> dict[str, Any]:
        """Flatten one chunk record into a vector-store-friendly metadata payload."""

        sparse_vector = dict(record.sparse_vector)
        metadata = dict(record.metadata)
        payload = asdict(record) if hasattr(record, "__dataclass_fields__") else dict(record)
        return {
            "chunk_id": str(record.chunk_id),
            "document_id": str(record.document_id),
            "content_hash": str(record.content_hash),
            "source_path": str(record.source_path),
            "chunk_index": int(record.chunk_index),
            "source_ref": str(metadata.get("source_ref") or ""),
            "has_images": bool(record.image_refs),
            "image_refs_json": json.dumps(list(record.image_refs), ensure_ascii=False, sort_keys=True),
            "metadata_json": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            "sparse_vector_json": json.dumps(sparse_vector, ensure_ascii=False, sort_keys=True),
            "payload_json": json.dumps(payload, ensure_ascii=False, sort_keys=True),
            "document_length": int(sparse_vector.get("document_length", 0) or 0),
            "unique_terms": int(sparse_vector.get("unique_terms", 0) or 0),
        }
