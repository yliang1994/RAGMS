from __future__ import annotations

from ragms.runtime.settings_models import AppSettings, EvaluationSettings, resolve_evaluation_backends


def test_resolve_evaluation_backends_preserves_order_and_deduplicates() -> None:
    settings = AppSettings(evaluation=EvaluationSettings(backends=["custom_metrics", "ragas", "custom_metrics", "deepeval"]))

    assert resolve_evaluation_backends(settings) == ["custom_metrics", "ragas", "deepeval"]


def test_resolve_evaluation_backends_supports_mapping_config() -> None:
    assert resolve_evaluation_backends({"backends": ["ragas", "deepeval"]}) == ["ragas", "deepeval"]
    assert resolve_evaluation_backends({"provider": "custom_metrics"}) == ["custom_metrics"]
