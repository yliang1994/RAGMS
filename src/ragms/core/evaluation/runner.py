"""Composite evaluator orchestration for multi-backend local evaluation."""

from __future__ import annotations

from typing import Any

from ragms.libs.abstractions import BaseEvaluator
from ragms.libs.factories.evaluator_factory import EvaluatorFactory
from ragms.runtime.settings_models import AppSettings, EvaluationSettings, resolve_evaluation_backends


class CompositeEvaluator(BaseEvaluator):
    """Execute multiple evaluators and normalize their combined result."""

    def __init__(self, evaluators: dict[str, BaseEvaluator]) -> None:
        self.evaluators = dict(evaluators)

    def evaluate(
        self,
        predictions: list[str],
        references: list[str] | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run all configured evaluators and aggregate their backend metrics."""

        backend_results: dict[str, dict[str, Any]] = {}
        sample_errors: list[dict[str, Any]] = []
        aggregate_buckets: dict[str, list[float]] = {}
        for backend_name, evaluator in self.evaluators.items():
            result = evaluator.evaluate(predictions, references, metadata=metadata)
            backend_results[backend_name] = result
            for error in result.get("errors") or []:
                sample_errors.append(dict(error))
            if result.get("status") != "succeeded":
                continue
            for metric_name, metric_value in (result.get("metrics") or {}).items():
                aggregate_buckets.setdefault(metric_name, []).append(float(metric_value))

        aggregate_metrics = {
            metric_name: sum(values) / len(values)
            for metric_name, values in aggregate_buckets.items()
            if values
        }
        return {
            "aggregate_metrics": aggregate_metrics,
            "backend_results": backend_results,
            "sample_errors": sample_errors,
        }


def build_evaluator_stack(
    config: AppSettings | EvaluationSettings | dict[str, Any] | None,
) -> CompositeEvaluator:
    """Build a composite evaluator from configured backend order."""

    evaluators = EvaluatorFactory.create_stack(config)
    return CompositeEvaluator(evaluators)


def resolve_evaluator_backend_set(
    config: AppSettings | EvaluationSettings | dict[str, Any] | None,
) -> list[str]:
    """Resolve the configured backend names in execution order."""

    return resolve_evaluation_backends(config)
