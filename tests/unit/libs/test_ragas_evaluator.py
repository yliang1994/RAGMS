from __future__ import annotations

from ragms.libs.providers.evaluators.ragas_evaluator import RagasEvaluator


def test_ragas_evaluator_returns_normalized_metrics_when_backend_is_available() -> None:
    evaluator = RagasEvaluator()

    result = evaluator.evaluate(
        ["answer"],
        ["answer"],
        metadata={"evaluation_modes": ["answer"], "allow_missing_backend_stub": True},
    )

    assert result["status"] == "succeeded"
    assert result["metrics"]["context_precision"] == 1.0
    assert result["metrics"]["faithfulness"] == 1.0


def test_ragas_evaluator_skips_when_answer_metrics_do_not_apply() -> None:
    evaluator = RagasEvaluator()

    result = evaluator.evaluate(
        ["answer"],
        ["answer"],
        metadata={"evaluation_modes": ["retrieval"]},
    )

    assert result["status"] == "skipped"
    assert result["raw_summary"]["skip_reason"] == "answer_metrics_not_applicable"
