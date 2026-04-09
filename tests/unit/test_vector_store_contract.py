from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ragms.libs.abstractions import BaseVectorStore
from ragms.libs.providers.vector_stores.chroma_store import ChromaStore, VectorStoreProviderError


def _build_store(tmp_path: Path) -> BaseVectorStore:
    return ChromaStore(collection="contract-docs", persist_directory=str(tmp_path / "chroma"))


class _ExplodingCollection:
    def __init__(self, *, action: str) -> None:
        self.action = action

    def upsert(self, **_: Any) -> None:
        if self.action == "add":
            raise RuntimeError("upsert exploded")

    def count(self) -> int:
        return 1

    def query(self, **_: Any) -> dict[str, list[list[Any]]]:
        if self.action == "query":
            raise RuntimeError("query exploded")
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    def get(self, **_: Any) -> dict[str, list[str]]:
        return {"ids": ["doc-1"]}

    def delete(self, **_: Any) -> dict[str, int]:
        if self.action == "delete":
            raise RuntimeError("delete exploded")
        return {"deleted": 1}


class _FakeClient:
    def __init__(self, collection: _ExplodingCollection) -> None:
        self.collection = collection

    def get_or_create_collection(self, *, name: str, embedding_function: Any = None) -> _ExplodingCollection:
        del name, embedding_function
        return self.collection


@pytest.mark.unit
def test_vector_store_contract_round_trip_delete_and_result_shape(tmp_path: Path) -> None:
    store = _build_store(tmp_path)

    added_ids = store.add(
        ids=["chunk-a", "chunk-b"],
        vectors=[[0.0, 0.0], [1.0, 1.0]],
        documents=["alpha", "beta"],
        metadatas=[{"topic": "finance"}, {"topic": "ops"}],
    )
    matches = store.query([0.1, 0.1], top_k=2)
    deleted = store.delete(["chunk-a"])
    remaining = store.query([0.1, 0.1], top_k=2)

    assert added_ids == ["chunk-a", "chunk-b"]
    assert [match["id"] for match in matches] == ["chunk-a", "chunk-b"]
    assert list(matches[0]) == ["id", "score", "document", "metadata"]
    assert matches[0]["document"] == "alpha"
    assert matches[0]["metadata"] == {"topic": "finance"}
    assert matches[0]["score"] > matches[1]["score"]
    assert deleted == 1
    assert [match["id"] for match in remaining] == ["chunk-b"]


@pytest.mark.unit
def test_vector_store_contract_supports_metadata_filters_and_empty_cases(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    store.add(
        ids=["chunk-a", "chunk-b"],
        vectors=[[0.0, 0.0], [0.1, 0.1]],
        documents=["alpha", "beta"],
        metadatas=[{"topic": "finance"}, {"topic": "ops"}],
    )

    filtered = store.query([0.0, 0.0], top_k=5, filters={"topic": "ops"})

    assert [match["id"] for match in filtered] == ["chunk-b"]
    assert filtered[0]["metadata"]["topic"] == "ops"
    assert store.query([0.0, 0.0], top_k=0) == []
    assert store.delete(["missing"]) == 0


@pytest.mark.unit
def test_vector_store_contract_rejects_misaligned_inputs(tmp_path: Path) -> None:
    store = _build_store(tmp_path)

    with pytest.raises(ValueError, match="ids and vectors must have the same length"):
        store.add(ids=["chunk-a"], vectors=[[0.0, 0.0], [1.0, 1.0]])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("action", "message"),
    [
        ("add", "Chroma add request failed"),
        ("query", "Chroma query request failed"),
        ("delete", "Chroma delete request failed"),
    ],
)
def test_vector_store_contract_wraps_provider_failures(action: str, message: str) -> None:
    store = ChromaStore(collection="contract-docs", client=_FakeClient(_ExplodingCollection(action=action)))

    with pytest.raises(VectorStoreProviderError, match=message):
        if action == "add":
            store.add(ids=["chunk-a"], vectors=[[0.0, 0.0]])
        elif action == "query":
            store.query([0.0, 0.0], top_k=1)
        else:
            store.delete(["chunk-a"])
