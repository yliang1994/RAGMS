"""Deterministic retrieval metrics used by local evaluation and regression gates."""

from __future__ import annotations

import math


def compute_hit_rate_at_k(retrieved_ids: list[str], expected_ids: list[str], *, k: int) -> float:
    """Return whether any expected id appears within the top-k retrieved ids."""

    if k <= 0:
        raise ValueError("k must be greater than zero")
    if not expected_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return 1.0 if any(item in top_k for item in expected_ids) else 0.0


def compute_mrr(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    """Return mean reciprocal rank for one sample."""

    if not expected_ids:
        return 0.0
    expected = set(expected_ids)
    for index, item in enumerate(retrieved_ids, start=1):
        if item in expected:
            return 1.0 / float(index)
    return 0.0


def compute_ndcg_at_k(retrieved_ids: list[str], expected_ids: list[str], *, k: int) -> float:
    """Return normalized discounted cumulative gain at k."""

    if k <= 0:
        raise ValueError("k must be greater than zero")
    expected = set(expected_ids)
    if not expected:
        return 0.0

    dcg = 0.0
    for index, item in enumerate(retrieved_ids[:k], start=1):
        if item in expected:
            dcg += 1.0 / math.log2(index + 1)

    ideal_hits = min(len(expected), k)
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg
