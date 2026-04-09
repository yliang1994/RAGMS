from __future__ import annotations

import pytest

from ragms.core.evaluation import DatasetLoader
from ragms.core.evaluation.metrics.answer_metrics import compute_citation_coverage
from ragms.core.evaluation.metrics.retrieval_metrics import compute_hit_rate_at_k, compute_mrr


def assert_recall_thresholds(metrics: dict[str, float], thresholds: dict[str, float]) -> None:
    """Assert deterministic retrieval thresholds with explicit failure messages."""

    failures = [
        f"{name} expected >= {minimum}, got {metrics.get(name)}"
        for name, minimum in thresholds.items()
        if float(metrics.get(name, 0.0)) < float(minimum)
    ]
    if failures:
        raise AssertionError("; ".join(failures))


@pytest.mark.e2e
def test_recall_regression() -> None:
    dataset = DatasetLoader("data/evaluation/datasets").load(dataset_name="golden", dataset_version="v1")
    sample = dataset["samples"][0]

    retrieved_ids = ["chunk_cfg_001", "chunk_other_002"]
    answer = "使用 settings.yaml 配置 endpoint、api_key 与 deployment [1]."
    citations = [{"index": 1, "chunk_id": "chunk_cfg_001"}]
    metrics = {
        "hit_rate_at_k": compute_hit_rate_at_k(retrieved_ids, sample.expected_chunk_ids, k=2),
        "mrr": compute_mrr(retrieved_ids, sample.expected_chunk_ids),
        "citation_coverage": compute_citation_coverage(answer, citations),
    }

    assert_recall_thresholds(
        metrics,
        {
            "hit_rate_at_k": 1.0,
            "mrr": 1.0,
            "citation_coverage": 1.0,
        },
    )
