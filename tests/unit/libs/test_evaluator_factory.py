from __future__ import annotations

import pytest

from ragms.libs.factories.evaluator_factory import EvaluatorFactory
from ragms.libs.providers.evaluators.custom_metrics_evaluator import CustomMetricsEvaluator
from ragms.libs.providers.evaluators.ragas_evaluator import RagasEvaluator


def test_evaluator_factory_creates_configured_evaluator() -> None:
    evaluator = EvaluatorFactory.create({"type": "ragas", "metrics": ["context_precision"]})
    assert isinstance(evaluator, RagasEvaluator)

    result = evaluator.evaluate([{"expected": "a", "actual": "a"}])
    assert result["backend"] == "ragas"
    assert isinstance(result["metrics"], dict)


def test_evaluator_factory_supports_custom_metrics_output_shape() -> None:
    evaluator = EvaluatorFactory.create({"type": "custom_metrics", "metrics": ["hit_rate"]})
    assert isinstance(evaluator, CustomMetricsEvaluator)

    result = evaluator.evaluate(
        [
            {"expected": "yes", "actual": "yes"},
            {"expected": "no", "actual": "yes"},
        ]
    )
    assert result["backend"] == "custom_metrics"
    assert result["metrics"] == {"hit_rate": 0.5}
    assert result["sample_count"] == 2


def test_evaluator_factory_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="requires type"):
        EvaluatorFactory.create({})

    with pytest.raises(ValueError, match="Unknown evaluator type"):
        EvaluatorFactory.create({"type": "missing"})
