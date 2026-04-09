"""Aggregation helpers for evaluation report metrics."""

from __future__ import annotations

from typing import Any, Mapping


def aggregate_evaluation_metrics(
    samples: list[Mapping[str, Any]],
    *,
    failed_samples: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Average numeric sample metrics and attach stable run counters."""

    buckets: dict[str, list[float]] = {}
    for sample in samples:
        metrics = dict(sample.get("metrics_summary") or {})
        for name, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            buckets.setdefault(str(name), []).append(float(value))

    aggregate = {
        name: sum(values) / len(values)
        for name, values in buckets.items()
        if values
    }
    failed_count = len(failed_samples or [])
    aggregate["sample_count"] = len(samples) + failed_count
    aggregate["succeeded_sample_count"] = len(samples)
    aggregate["failed_sample_count"] = failed_count
    return aggregate
