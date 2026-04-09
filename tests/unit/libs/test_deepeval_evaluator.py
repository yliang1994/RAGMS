from __future__ import annotations

from ragms.libs.providers.evaluators.deepeval_evaluator import DeepEvalEvaluator


def test_deepeval_evaluator_returns_normalized_metrics_when_backend_is_available() -> None:
    evaluator = DeepEvalEvaluator()

    result = evaluator.evaluate(
        ["answer"],
        ["answer"],
        metadata={"evaluation_modes": ["answer"], "allow_missing_backend_stub": True},
    )

    assert result["status"] == "succeeded"
    assert result["metrics"]["answer_relevancy"] == 1.0
    assert result["metrics"]["correctness"] == 1.0


def test_deepeval_evaluator_skips_when_dependency_is_missing() -> None:
    evaluator = DeepEvalEvaluator()

    result = evaluator.evaluate(
        ["answer"],
        ["answer"],
        metadata={"evaluation_modes": ["answer"]},
    )

    assert result["status"] == "skipped"
    assert result["raw_summary"]["skip_reason"] == "dependency_missing"
