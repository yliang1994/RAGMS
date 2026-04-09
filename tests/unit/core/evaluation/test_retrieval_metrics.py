from __future__ import annotations

import pytest

from ragms.core.evaluation.metrics.retrieval_metrics import (
    compute_hit_rate_at_k,
    compute_mrr,
    compute_ndcg_at_k,
)


def test_retrieval_metrics_compute_expected_scores() -> None:
    retrieved = ["chunk-2", "chunk-1", "chunk-9"]
    expected = ["chunk-1"]

    assert compute_hit_rate_at_k(retrieved, expected, k=2) == 1.0
    assert compute_mrr(retrieved, expected) == 0.5
    assert 0.0 < compute_ndcg_at_k(retrieved, expected, k=3) <= 1.0


def test_retrieval_metrics_handle_missing_expected_items() -> None:
    assert compute_hit_rate_at_k(["chunk-2"], [], k=1) == 0.0
    assert compute_mrr(["chunk-2"], ["chunk-1"]) == 0.0
    assert compute_ndcg_at_k(["chunk-2"], ["chunk-1"], k=1) == 0.0


@pytest.mark.parametrize("function", [compute_hit_rate_at_k, compute_ndcg_at_k])
def test_retrieval_metrics_reject_non_positive_k(function) -> None:
    with pytest.raises(ValueError, match="k must be greater than zero"):
        function(["chunk-1"], ["chunk-1"], k=0)
