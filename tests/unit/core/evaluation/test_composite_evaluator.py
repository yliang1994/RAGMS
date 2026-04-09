from __future__ import annotations

from ragms.core.evaluation.runner import CompositeEvaluator, build_evaluator_stack, resolve_evaluator_backend_set
from ragms.libs.abstractions import BaseEvaluator
from ragms.runtime.settings_models import AppSettings, EvaluationSettings


class _SucceededEvaluator(BaseEvaluator):
    def __init__(self, score: float) -> None:
        self.score = score

    def evaluate(self, predictions, references=None, *, metadata=None):  # noqa: ANN001
        return {
            "status": "succeeded",
            "metrics": {"score": self.score},
            "errors": [],
            "raw_summary": {},
        }


class _SkippedEvaluator(BaseEvaluator):
    def evaluate(self, predictions, references=None, *, metadata=None):  # noqa: ANN001
        return {
            "status": "skipped",
            "metrics": {},
            "errors": [{"backend": "skip", "message": "not_applicable"}],
            "raw_summary": {"skip_reason": "not_applicable"},
        }


def test_composite_evaluator_aggregates_backend_metrics_and_errors() -> None:
    evaluator = CompositeEvaluator(
        {
            "custom_metrics": _SucceededEvaluator(0.8),
            "ragas": _SucceededEvaluator(0.6),
            "deepeval": _SkippedEvaluator(),
        }
    )

    result = evaluator.evaluate(["answer"], ["reference"])

    assert result["aggregate_metrics"]["score"] == 0.7
    assert result["backend_results"]["deepeval"]["status"] == "skipped"
    assert result["sample_errors"][0]["message"] == "not_applicable"


def test_build_evaluator_stack_and_backend_resolution_follow_config_order() -> None:
    settings = AppSettings(evaluation=EvaluationSettings(backends=["custom_metrics", "ragas", "deepeval"]))

    stack = build_evaluator_stack(settings)
    backend_set = resolve_evaluator_backend_set(settings)

    assert list(stack.evaluators.keys()) == ["custom_metrics", "ragas", "deepeval"]
    assert backend_set == ["custom_metrics", "ragas", "deepeval"]
