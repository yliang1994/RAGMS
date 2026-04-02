from __future__ import annotations

from pathlib import Path

import pytest

from ragms.libs.providers.vector_stores.chroma_store import ChromaStore


def test_chroma_store_add_query_and_delete_round_trip(tmp_path: Path) -> None:
    store = ChromaStore(collection="docs", persist_directory=str(tmp_path / "chroma"))

    added_ids = store.add(
        ids=["a", "b"],
        vectors=[[0.0, 0.0], [1.0, 1.0]],
        documents=["first doc", "second doc"],
        metadatas=[{"topic": "alpha"}, {"topic": "beta"}],
    )
    matches = store.query([0.1, 0.1], top_k=2)
    deleted = store.delete(["a"])
    remaining = store.query([0.1, 0.1], top_k=2)

    assert added_ids == ["a", "b"]
    assert [match["id"] for match in matches] == ["a", "b"]
    assert matches[0]["document"] == "first doc"
    assert matches[0]["metadata"]["topic"] == "alpha"
    assert matches[0]["score"] > matches[1]["score"]
    assert deleted == 1
    assert [match["id"] for match in remaining] == ["b"]


def test_chroma_store_query_respects_filters(tmp_path: Path) -> None:
    store = ChromaStore(collection="docs", persist_directory=str(tmp_path / "chroma"))
    store.add(
        ids=["a", "b"],
        vectors=[[0.0, 0.0], [0.1, 0.1]],
        documents=["alpha", "beta"],
        metadatas=[{"topic": "finance"}, {"topic": "ops"}],
    )

    matches = store.query([0.0, 0.0], top_k=5, filters={"topic": "ops"})

    assert [match["id"] for match in matches] == ["b"]
    assert matches[0]["metadata"]["topic"] == "ops"


def test_chroma_store_returns_empty_results_for_empty_collection(tmp_path: Path) -> None:
    store = ChromaStore(collection="docs", persist_directory=str(tmp_path / "chroma"))

    assert store.query([0.0, 0.0], top_k=3) == []


def test_chroma_store_returns_zero_for_missing_ids(tmp_path: Path) -> None:
    store = ChromaStore(collection="docs", persist_directory=str(tmp_path / "chroma"))
    store.add(ids=["a"], vectors=[[0.0, 0.0]])

    assert store.delete(["missing"]) == 0


def test_chroma_store_rejects_mismatched_input_lengths(tmp_path: Path) -> None:
    store = ChromaStore(collection="docs", persist_directory=str(tmp_path / "chroma"))

    with pytest.raises(ValueError, match="ids and vectors must have the same length"):
        store.add(ids=["a"], vectors=[[0.0, 0.0], [1.0, 1.0]])
