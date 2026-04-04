"""Storage pipeline that assembles chunk records before vector upsert."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from ragms.ingestion_pipeline.storage.vector_upsert import VectorUpsert


@dataclass(frozen=True)
class ChunkRecord:
    """Serializable storage record for one enriched chunk."""

    chunk_id: str
    document_id: str
    content: str
    metadata: dict[str, Any]
    dense_vector: list[float]
    sparse_vector: dict[str, Any]
    content_hash: str
    source_path: str
    chunk_index: int
    image_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the record into a dict/json-friendly payload."""

        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content": self.content,
            "metadata": dict(self.metadata),
            "dense_vector": list(self.dense_vector),
            "sparse_vector": dict(self.sparse_vector),
            "content_hash": self.content_hash,
            "source_path": self.source_path,
            "chunk_index": self.chunk_index,
            "image_refs": list(self.image_refs),
        }


class ChunkRecordBuilder:
    """Build stable chunk records from chunk payloads plus encoded vectors."""

    def build(
        self,
        chunks: list[dict[str, Any]],
        *,
        dense_vectors: list[list[float]],
        sparse_vectors: list[dict[str, Any]],
    ) -> list[ChunkRecord]:
        """Assemble chunk records with aligned dense and sparse payloads."""

        if len(chunks) != len(dense_vectors) or len(chunks) != len(sparse_vectors):
            raise ValueError("chunks, dense_vectors, and sparse_vectors must have the same length")

        records: list[ChunkRecord] = []
        for chunk, dense_vector, sparse_vector in zip(chunks, dense_vectors, sparse_vectors, strict=True):
            content = str(chunk.get("content", ""))
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            metadata = dict(chunk.get("metadata") or {})
            image_refs = list(chunk.get("image_refs") or metadata.get("image_refs") or [])
            record = ChunkRecord(
                chunk_id=str(chunk.get("chunk_id", "")),
                document_id=str(chunk.get("document_id", metadata.get("document_id", ""))),
                content=content,
                metadata=metadata,
                dense_vector=[float(value) for value in dense_vector],
                sparse_vector=dict(sparse_vector),
                content_hash=content_hash,
                source_path=str(chunk.get("source_path", "")),
                chunk_index=int(chunk.get("chunk_index", metadata.get("chunk_index", 0) or 0)),
                image_refs=image_refs,
            )
            if not record.chunk_id:
                raise ValueError("chunk_id is required for storage records")
            if not record.document_id:
                raise ValueError("document_id is required for storage records")
            records.append(record)
        return records


class StoragePipeline:
    """Coordinate chunk-record assembly and vector-store persistence."""

    def __init__(
        self,
        *,
        vector_upsert: VectorUpsert,
        record_builder: ChunkRecordBuilder | None = None,
    ) -> None:
        self.vector_upsert = vector_upsert
        self.record_builder = record_builder or ChunkRecordBuilder()

    def run(
        self,
        chunks: list[dict[str, Any]],
        *,
        dense_vectors: list[list[float]],
        sparse_vectors: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build storage records, write them, and return a stable summary payload."""

        records = self.record_builder.build(
            chunks,
            dense_vectors=dense_vectors,
            sparse_vectors=sparse_vectors,
        )
        written_ids = self.vector_upsert.write(records)
        return {
            "chunk_records": records,
            "records": [record.to_dict() for record in records],
            "record_count": len(records),
            "written_ids": list(written_ids),
            "written_count": len(written_ids),
        }
