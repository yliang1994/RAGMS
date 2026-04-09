from __future__ import annotations

from ragms.core.evaluation.metrics.answer_metrics import (
    compute_answer_structure_score,
    compute_citation_coverage,
)


def test_answer_metrics_compute_citation_coverage() -> None:
    answer = "RAG combines retrieval and generation [1]."
    citations = [{"index": 1}, {"index": 2}]

    assert compute_citation_coverage(answer, citations) == 0.5


def test_answer_metrics_compute_structure_score() -> None:
    assert compute_answer_structure_score("RAG combines retrieval and generation [1].") == 1.0
    assert compute_answer_structure_score(" ") == 0.0
