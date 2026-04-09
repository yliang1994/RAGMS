from __future__ import annotations

from typing import Any

import pytest

from ragms.core.evaluation.runner import CompositeEvaluator
from ragms.libs.abstractions import BaseEvaluator
from ragms.libs.abstractions.base_evaluator import normalize_backend_metrics, serialize_backend_failure


class _ContractEvaluator(BaseEvaluator):
    def __init__(
        self,
        *,
        status: str = "succeeded",
        metrics: dict[str, float] | None = None,
        errors: list[dict[str, Any]] | None = None,
        raw_summary: dict[str, Any] | None = None,
    ) -> None:
        self.status = status
        self.metrics = dict(metrics or {})
        self.errors = list(errors or [])
        self.raw_summary = dict(raw_summary or {})
        self.calls: list[dict[str, Any]] = []

    def evaluate(
        self,
        predictions: list[str],
        references: list[str] | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "predictions": list(predictions),
                "references": list(references or []),
                "metadata": dict(metadata or {}),
            }
        )
        return normalize_backend_metrics(
            status=self.status,
            metrics=self.metrics,
            errors=self.errors,
            raw_summary=self.raw_summary,
        )


@pytest.mark.unit
def test_base_evaluator_contract_normalizes_backend_result_shape() -> None:
    evaluator = _ContractEvaluator(
        metrics={"hit_rate": 0.9},
        raw_summary={"backend_latency_ms": 12.5},
    )

    result = evaluator.evaluate(
        ["answer"],
        ["reference"],
        metadata={"retrieved_ids": ["chunk-a"]},
    )

    assert list(result) == ["status", "metrics", "errors", "raw_summary"]
    assert result["status"] == "succeeded"
    assert result["metrics"] == {"hit_rate": 0.9}
    assert result["errors"] == []
    assert result["raw_summary"]["backend_latency_ms"] == 12.5
    assert evaluator.calls[0]["metadata"]["retrieved_ids"] == ["chunk-a"]


@pytest.mark.unit
def test_composite_evaluator_contract_aggregates_successes_and_converges_backend_failures() -> None:
    failure = serialize_backend_failure("ragas", message="dependency missing", failure_type="skipped")
    evaluator = CompositeEvaluator(
        {
            "custom_metrics": _ContractEvaluator(metrics={"hit_rate": 0.9, "mrr": 0.8}),
            "ragas": _ContractEvaluator(
                status="skipped",
                errors=[failure],
                raw_summary={"skip_reason": "dependency_missing"},
            ),
            "plain_mapping": _ContractEvaluator(metrics={"hit_rate": 0.7}),
        }
    )

    result = evaluator.evaluate(["answer"], ["reference"], metadata={"sample_id": "sample-1"})

    assert result["aggregate_metrics"] == {"hit_rate": pytest.approx(0.8), "mrr": pytest.approx(0.8)}
    assert result["backend_results"]["custom_metrics"]["status"] == "succeeded"
    assert result["backend_results"]["ragas"]["status"] == "skipped"
    assert result["backend_results"]["ragas"]["errors"][0]["backend"] == "ragas"
    assert result["backend_results"]["ragas"]["raw_summary"]["skip_reason"] == "dependency_missing"
    assert result["sample_errors"] == [failure]


@pytest.mark.unit
def test_composite_evaluator_contract_accepts_plain_metric_mappings() -> None:
    class _PlainMetricsEvaluator(BaseEvaluator):
        def evaluate(self, predictions, references=None, *, metadata=None):  # noqa: ANN001
            del predictions, references, metadata
            return {"faithfulness": 0.92}

    result = CompositeEvaluator({"custom_metrics": _PlainMetricsEvaluator()}).evaluate(["answer"], ["reference"])

    assert result["aggregate_metrics"] == {"faithfulness": 0.92}
    assert result["backend_results"]["custom_metrics"]["status"] == "succeeded"
    assert result["backend_results"]["custom_metrics"]["errors"] == []
