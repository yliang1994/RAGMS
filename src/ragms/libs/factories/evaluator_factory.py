from __future__ import annotations

from typing import Any

from ragms.libs.abstractions import BaseEvaluator
from ragms.libs.providers.evaluators.custom_metrics_evaluator import CustomMetricsEvaluator
from ragms.libs.providers.evaluators.deepeval_evaluator import DeepEvalEvaluator
from ragms.libs.providers.evaluators.ragas_evaluator import RagasEvaluator


class EvaluatorFactory:
    _REGISTRY = {
        "ragas": RagasEvaluator,
        "deepeval": DeepEvalEvaluator,
        "custom_metrics": CustomMetricsEvaluator,
    }

    @classmethod
    def create(cls, config: dict[str, Any]) -> BaseEvaluator:
        evaluator_type = config.get("type")
        if not evaluator_type:
            raise ValueError("Evaluator config requires type")
        try:
            evaluator_class = cls._REGISTRY[evaluator_type]
        except KeyError as exc:
            raise ValueError(f"Unknown evaluator type: {evaluator_type}") from exc
        return evaluator_class(metrics=config.get("metrics"))

