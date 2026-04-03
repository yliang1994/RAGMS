from __future__ import annotations

import json

import pytest

from ragms.ingestion_pipeline.storage import ChunkRecordBuilder, StoragePipeline, VectorUpsert
from tests.fakes.fake_vector_store import FakeVectorStore


def _chunk() -> dict[str, object]:
    return {
        "chunk_id": "doc-1_0000_abc12345_def678",
        "document_id": "doc-1",
        "content": "Alpha chunk with image context",
        "source_path": "docs/alpha.pdf",
        "chunk_index": 0,
        "image_refs": ["img-1"],
        "metadata": {
            "document_id": "doc-1",
            "source_ref": "p1",
            "image_occurrences": [{"image_id": "img-1", "text_offset": 5, "text_length": 10}],
            "images": [{"id": "img-1", "page": 1, "path": "data/images/demo/img-1.png"}],
            "chunk_title": "Alpha",
        },
    }


def _sparse() -> dict[str, object]:
    return {
        "content_hash": "ignored-by-builder",
        "tokens": ["alpha", "chunk"],
        "term_frequencies": {"alpha": 1, "chunk": 1},
        "term_weights": {"alpha": 0.5, "chunk": 0.5},
        "document_length": 2,
        "unique_terms": 2,
    }


def test_chunk_record_builder_builds_stable_records() -> None:
    builder = ChunkRecordBuilder()

    records = builder.build(
        [_chunk()],
        dense_vectors=[[0.1, 0.2]],
        sparse_vectors=[_sparse()],
    )

    assert len(records) == 1
    record = records[0]
    assert record.chunk_id == "doc-1_0000_abc12345_def678"
    assert record.document_id == "doc-1"
    assert record.dense_vector == [0.1, 0.2]
    assert record.sparse_vector["tokens"] == ["alpha", "chunk"]
    assert record.image_refs == ["img-1"]
    assert record.metadata["images"][0]["id"] == "img-1"
    assert len(record.content_hash) == 64


def test_storage_pipeline_writes_records_and_preserves_image_metadata() -> None:
    vector_store = FakeVectorStore()
    pipeline = StoragePipeline(vector_upsert=VectorUpsert(vector_store))

    result = pipeline.run(
        [_chunk()],
        dense_vectors=[[0.1, 0.2]],
        sparse_vectors=[_sparse()],
    )

    assert result["record_count"] == 1
    assert result["written_count"] == 1
    stored = vector_store._items[result["written_ids"][0]]
    metadata_json = json.loads(stored.metadata["metadata_json"])
    payload_json = json.loads(stored.metadata["payload_json"])
    assert metadata_json["images"][0]["id"] == "img-1"
    assert payload_json["image_refs"] == ["img-1"]
    assert stored.metadata["has_images"] is True


def test_storage_pipeline_rejects_mismatched_lengths() -> None:
    pipeline = StoragePipeline(vector_upsert=VectorUpsert(FakeVectorStore()))

    with pytest.raises(
        ValueError,
        match="chunks, dense_vectors, and sparse_vectors must have the same length",
    ):
        pipeline.run(
            [_chunk()],
            dense_vectors=[[0.1, 0.2], [0.3, 0.4]],
            sparse_vectors=[_sparse()],
        )
