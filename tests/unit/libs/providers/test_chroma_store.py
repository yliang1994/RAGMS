from __future__ import annotations

from ragms.libs.providers.vector_stores.chroma_store import ChromaStore


def test_chroma_store_adds_and_queries_vectors() -> None:
    store = ChromaStore(path="data/vector_store/chroma", collection_prefix="ragms_", collection_name="demo")

    added = store.add(
        [
            {"id": "a", "text": "ragms retrieval", "embedding": [14.0, 5.0, 0.0]},
            {"id": "b", "text": "other topic", "embedding": [5.0, 1.0, 0.0]},
        ]
    )
    results = store.query("ragms retrieval", top_k=1)

    assert added == 2
    assert results[0]["id"] == "a"
    assert store.full_collection_name == "ragms_demo"


def test_chroma_store_delete_and_missing_ids_behavior() -> None:
    store = ChromaStore(path="data/vector_store/chroma")
    store.add([{"id": "a", "text": "ragms retrieval", "embedding": [1.0, 0.0, 0.0]}])

    deleted = store.delete(["a", "missing"])

    assert deleted == 1
    assert store.query("ragms retrieval") == []


def test_chroma_store_handles_empty_collection() -> None:
    store = ChromaStore(path="data/vector_store/chroma")

    assert store.query("anything") == []
