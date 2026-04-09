"""Lightweight Ragas-compatible evaluator stub for factory wiring."""

from __future__ import annotations

import importlib.util
from typing import Any

from ragms.libs.abstractions import BaseEvaluator, normalize_backend_metrics, serialize_backend_failure


class RagasEvaluator(BaseEvaluator):
    """Return standardized Ragas-style metrics without importing ragas yet."""

    def evaluate(
        self,
        predictions: list[str],
        references: list[str] | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return normalized ragas-compatible metrics with structured fallback behavior."""

        metadata = dict(metadata or {})
        evaluation_modes = set(metadata.get("evaluation_modes") or ["answer"])
        if "answer" not in evaluation_modes:
            return normalize_backend_metrics(
                status="skipped",
                errors=[serialize_backend_failure("ragas", message="answer_metrics_not_applicable", failure_type="skip_reason")],
                raw_summary={"skip_reason": "answer_metrics_not_applicable"},
            )
        if importlib.util.find_spec("ragas") is None and not metadata.get("allow_missing_backend_stub", False):
            return normalize_backend_metrics(
                status="skipped",
                errors=[serialize_backend_failure("ragas", message="ragas dependency is not installed", failure_type="dependency_missing")],
                raw_summary={"skip_reason": "dependency_missing"},
            )

        references = references or []
        pair_count = min(len(predictions), len(references))
        answer_relevancy = float(pair_count / len(predictions)) if predictions else 0.0
        faithfulness = float(
            sum(
                1
                for index in range(pair_count)
                if predictions[index].strip().casefold() == references[index].strip().casefold()
            )
            / pair_count
        ) if pair_count else 0.0
        return normalize_backend_metrics(
            status="succeeded",
            metrics={
                "context_precision": answer_relevancy,
                "answer_relevancy": answer_relevancy,
                "faithfulness": faithfulness,
            },
            raw_summary={"sample_count": len(predictions)},
        )
