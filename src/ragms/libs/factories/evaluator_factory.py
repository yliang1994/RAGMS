"""Factory for evaluator provider instantiation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ragms.libs.abstractions import BaseEvaluator
from ragms.libs.providers.evaluators.custom_metrics_evaluator import CustomMetricsEvaluator
from ragms.libs.providers.evaluators.deepeval_evaluator import DeepEvalEvaluator
from ragms.libs.providers.evaluators.ragas_evaluator import RagasEvaluator
from ragms.runtime.exceptions import RagMSError
from ragms.runtime.settings_models import AppSettings, EvaluationSettings


def _resolve_backend(backends: list[str] | tuple[str, ...] | None) -> str:
    """Resolve the first configured evaluation backend."""

    if not backends:
        raise RagMSError("Missing evaluator provider in configuration")
    provider = str(backends[0]).strip().lower()
    if not provider:
        raise RagMSError("Missing evaluator provider in configuration")
    return provider


def _coerce_evaluator_config(
    config: AppSettings | EvaluationSettings | Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Normalize evaluator configuration into factory keyword arguments."""

    if config is None:
        return {}
    if isinstance(config, AppSettings):
        return {"provider": _resolve_backend(config.evaluation.backends)}
    if isinstance(config, EvaluationSettings):
        return {"provider": _resolve_backend(config.backends)}
    options = dict(config)
    if "provider" in options:
        return options
    if "backend" in options:
        options["provider"] = options.pop("backend")
        return options
    if "backends" in options:
        options["provider"] = _resolve_backend(options.pop("backends"))
        return options
    return options


class EvaluatorFactory:
    """Create evaluator implementations from provider configuration."""

    _PROVIDERS = {
        "custom_metrics": CustomMetricsEvaluator,
        "ragas": RagasEvaluator,
        "deepeval": DeepEvalEvaluator,
    }

    @staticmethod
    def create(
        config: AppSettings | EvaluationSettings | Mapping[str, Any] | None = None,
    ) -> BaseEvaluator:
        """Return an evaluator implementation for the configured provider."""

        options = _coerce_evaluator_config(config)
        if isinstance(config, Mapping) and "provider" not in options:
            raise RagMSError("Missing evaluator provider in configuration")

        provider = str(options.pop("provider", "custom_metrics")).strip().lower() or "custom_metrics"
        provider_class = EvaluatorFactory._PROVIDERS.get(provider)
        if provider_class is None:
            raise RagMSError(f"Unknown evaluator provider: {provider}")
        return provider_class(**options)
