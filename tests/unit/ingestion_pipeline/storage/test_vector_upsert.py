from __future__ import annotations

import json

import pytest

from ragms.ingestion_pipeline.storage import ChunkRecord, VectorUpsert, VectorUpsertError


class RecordingVectorStore:
    def __init__(self) -> None:
        self.add_calls: list[dict[str, object]] = []
        self.delete_calls: list[list[str]] = []
        self.items: dict[str, dict[str, object]] = {}

    def add(
        self,
        ids: list[str],
        vectors: list[list[float]],
        *,
        documents: list[str] | None = None,
        metadatas: list[dict[str, object]] | None = None,
    ) -> list[str]:
        payload = {
            "ids": list(ids),
            "vectors": [list(vector) for vector in vectors],
            "documents": list(documents or []),
            "metadatas": list(metadatas or []),
        }
        self.add_calls.append(payload)
        for index, item_id in enumerate(ids):
            self.items[item_id] = {
                "vector": list(vectors[index]),
                "document": (documents or [""] * len(ids))[index],
                "metadata": (metadatas or [{} for _ in ids])[index],
            }
        return ids

    def query(self, query_vector: list[float], *, top_k: int = 5, filters: dict[str, object] | None = None) -> list[dict[str, object]]:
        del query_vector, top_k, filters
        return []

    def delete(self, ids: list[str]) -> int:
        self.delete_calls.append(list(ids))
        for item_id in ids:
            self.items.pop(item_id, None)
        return len(ids)


class FailingVectorStore(RecordingVectorStore):
    def __init__(self, fail_on_call: int) -> None:
        super().__init__()
        self.fail_on_call = fail_on_call

    def add(
        self,
        ids: list[str],
        vectors: list[list[float]],
        *,
        documents: list[str] | None = None,
        metadatas: list[dict[str, object]] | None = None,
    ) -> list[str]:
        if len(self.add_calls) + 1 == self.fail_on_call:
            raise RuntimeError("boom")
        return super().add(ids, vectors, documents=documents, metadatas=metadatas)


def _record(
    *,
    chunk_id: str = "chunk-1",
    content: str = "alpha",
    content_hash: str = "hash-alpha",
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id="doc-1",
        content=content,
        metadata={
            "source_ref": "p1",
            "images": [{"id": "img-1", "page": 1}],
            "image_occurrences": [{"image_id": "img-1", "text_offset": 4, "text_length": 6}],
        },
        dense_vector=[0.1, 0.2],
        sparse_vector={
            "tokens": ["alpha"],
            "term_frequencies": {"alpha": 1},
            "term_weights": {"alpha": 1.0},
            "document_length": 1,
            "unique_terms": 1,
        },
        content_hash=content_hash,
        source_path="docs/a.pdf",
        chunk_index=0,
        image_refs=["img-1"],
    )


def test_vector_upsert_writes_stable_batches_with_flattened_metadata() -> None:
    store = RecordingVectorStore()
    upsert = VectorUpsert(store, batch_size=1)

    written_ids = upsert.write([_record(), _record(chunk_id="chunk-2", content_hash="hash-beta", content="beta")])

    assert written_ids == ["chunk-1", "chunk-2"]
    assert len(store.add_calls) == 2
    metadata = store.add_calls[0]["metadatas"][0]
    assert metadata["chunk_id"] == "chunk-1"
    assert metadata["has_images"] is True
    assert json.loads(metadata["metadata_json"])["images"][0]["id"] == "img-1"
    assert json.loads(metadata["sparse_vector_json"])["term_frequencies"]["alpha"] == 1


def test_vector_upsert_is_idempotent_for_repeated_same_content() -> None:
    store = RecordingVectorStore()
    upsert = VectorUpsert(store)
    record = _record()

    first = upsert.write([record])
    second = upsert.write([record])

    assert first == ["chunk-1"]
    assert second == []
    assert len(store.add_calls) == 1


def test_vector_upsert_rolls_back_previous_batches_on_failure() -> None:
    store = FailingVectorStore(fail_on_call=2)
    upsert = VectorUpsert(store, batch_size=1)

    with pytest.raises(VectorUpsertError, match="Vector upsert failed"):
        upsert.write([_record(), _record(chunk_id="chunk-2", content_hash="hash-beta", content="beta")])

    assert store.delete_calls == [["chunk-1"]]
    assert "chunk-1" not in store.items


def test_vector_upsert_rejects_conflicting_duplicate_ids() -> None:
    store = RecordingVectorStore()
    upsert = VectorUpsert(store)

    with pytest.raises(
        VectorUpsertError,
        match="Conflicting records share chunk_id chunk-1 but differ in content_hash",
    ):
        upsert.write([_record(), _record(content_hash="hash-other")])
