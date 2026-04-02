from __future__ import annotations

import pytest

from ragms.libs.factories.evaluator_factory import EvaluatorFactory
from ragms.libs.providers.evaluators.custom_metrics_evaluator import CustomMetricsEvaluator
from ragms.libs.providers.evaluators.deepeval_evaluator import DeepEvalEvaluator
from ragms.libs.providers.evaluators.ragas_evaluator import RagasEvaluator
from ragms.runtime.exceptions import RagMSError
from ragms.runtime.settings_models import AppSettings, EvaluationSettings


def test_evaluator_factory_uses_default_backend_from_app_settings() -> None:
    settings = AppSettings(evaluation=EvaluationSettings(backends=["custom_metrics"]))

    evaluator = EvaluatorFactory.create(settings)
    metrics = evaluator.evaluate(["answer"], ["answer"])

    assert isinstance(evaluator, CustomMetricsEvaluator)
    assert metrics == {
        "score": 1.0,
        "coverage": 1.0,
        "prediction_count": 1.0,
        "reference_count": 1.0,
    }


def test_evaluator_factory_uses_first_backend_from_evaluation_settings() -> None:
    evaluator = EvaluatorFactory.create(EvaluationSettings(backends=["ragas"]))

    assert isinstance(evaluator, RagasEvaluator)


def test_evaluator_factory_accepts_provider_mapping() -> None:
    evaluator = EvaluatorFactory.create({"provider": "deepeval"})

    assert isinstance(evaluator, DeepEvalEvaluator)


def test_evaluator_factory_accepts_backends_mapping() -> None:
    evaluator = EvaluatorFactory.create({"backends": ["custom_metrics", "ragas"]})

    assert isinstance(evaluator, CustomMetricsEvaluator)


def test_evaluator_factory_rejects_missing_provider_in_mapping() -> None:
    with pytest.raises(RagMSError, match="Missing evaluator provider in configuration"):
        EvaluatorFactory.create({"metrics": ["score"]})


def test_evaluator_factory_rejects_empty_backend_lists() -> None:
    with pytest.raises(RagMSError, match="Missing evaluator provider in configuration"):
        EvaluatorFactory.create(EvaluationSettings(backends=[]))


def test_evaluator_factory_rejects_unknown_provider() -> None:
    with pytest.raises(RagMSError, match="Unknown evaluator provider: custom"):
        EvaluatorFactory.create({"provider": "custom"})
