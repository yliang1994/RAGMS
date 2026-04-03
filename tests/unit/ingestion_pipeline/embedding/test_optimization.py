from __future__ import annotations

import time

import pytest

from ragms.ingestion_pipeline.embedding import optimize_embedding_batches


def test_optimize_embedding_batches_preserves_order_and_deduplicates_inputs() -> None:
    calls: list[list[str]] = []

    def process_batch(items: list[str]) -> list[dict[str, str]]:
        calls.append(list(items))
        return [{"value": item.upper()} for item in items]

    results = optimize_embedding_batches(
        ["alpha", "beta", "alpha", "gamma"],
        process_batch,
        batch_size=2,
    )

    assert calls == [["alpha", "beta"], ["gamma"]]
    assert results == [
        {"value": "ALPHA"},
        {"value": "BETA"},
        {"value": "ALPHA"},
        {"value": "GAMMA"},
    ]


def test_optimize_embedding_batches_reuses_cache() -> None:
    cache = {"alpha": {"value": "cached"}}
    calls: list[list[str]] = []

    def process_batch(items: list[str]) -> list[dict[str, str]]:
        calls.append(list(items))
        return [{"value": item.upper()} for item in items]

    results = optimize_embedding_batches(
        ["alpha", "beta"],
        process_batch,
        batch_size=2,
        cache=cache,
    )

    assert calls == [["beta"]]
    assert results == [{"value": "cached"}, {"value": "BETA"}]
    assert cache["beta"] == {"value": "BETA"}


def test_optimize_embedding_batches_preserves_semantics_with_parallel_batches() -> None:
    def process_batch(items: list[str]) -> list[str]:
        time.sleep(0.01 if items[0] == "gamma" else 0.0)
        return [item[::-1] for item in items]

    results = optimize_embedding_batches(
        ["alpha", "beta", "gamma", "delta"],
        process_batch,
        batch_size=1,
        max_workers=2,
    )

    assert results == ["ahpla", "ateb", "ammag", "atled"]


def test_optimize_embedding_batches_rejects_result_count_mismatch() -> None:
    def process_batch(items: list[str]) -> list[str]:
        return items[:1]

    with pytest.raises(ValueError, match="Processed batch size does not match input batch size"):
        optimize_embedding_batches(["alpha", "beta"], process_batch, batch_size=2)


def test_optimize_embedding_batches_rejects_invalid_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size must be greater than zero"):
        optimize_embedding_batches(["alpha"], lambda items: items, batch_size=0)
