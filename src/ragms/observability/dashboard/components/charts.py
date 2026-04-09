"""Shared metric-card and duration-chart helpers for dashboard pages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def render_metric_cards(
    metrics: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    renderer: Any | None = None,
) -> list[dict[str, Any]]:
    """Render or return normalized metric cards."""

    if isinstance(metrics, Mapping):
        cards = [
            {"label": str(key), "value": value}
            for key, value in metrics.items()
        ]
    else:
        cards = [
            {
                "label": str(item.get("label") or item.get("name") or "metric"),
                "value": item.get("value"),
                "delta": item.get("delta"),
                "description": item.get("description"),
            }
            for item in metrics
        ]
    if renderer is not None:
        for card in cards:
            renderer.metric(card["label"], card["value"], delta=card.get("delta"))
    return cards


def render_duration_chart(
    items: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    *,
    renderer: Any | None = None,
) -> dict[str, Any]:
    """Render or return a normalized duration series payload."""

    series = _build_duration_series(items)
    payload = {
        "points": series,
        "point_count": len(series),
        "max_duration_ms": max((point["duration_ms"] for point in series), default=0),
    }
    if renderer is not None and series:
        renderer.bar_chart(
            {
                point["label"]: point["duration_ms"]
                for point in series
            }
        )
    return payload


def _build_duration_series(items: Sequence[Mapping[str, Any]] | Mapping[str, Any]) -> list[dict[str, Any]]:
    if isinstance(items, Mapping) and "stage_comparisons" in items:
        points = []
        for comparison in items.get("stage_comparisons") or []:
            left = comparison.get("left") or {}
            right = comparison.get("right") or {}
            points.append(
                {
                    "label": str(comparison.get("stage_name") or "stage"),
                    "duration_ms": int(left.get("elapsed_ms") or 0),
                    "comparison_duration_ms": int(right.get("elapsed_ms") or 0),
                    "delta_ms": comparison.get("elapsed_delta_ms"),
                }
            )
        return points

    points = []
    for item in items:
        label = (
            item.get("stage_name")
            or item.get("trace_id")
            or item.get("label")
            or "item"
        )
        duration = item.get("elapsed_ms", item.get("duration_ms", 0))
        points.append(
            {
                "label": str(label),
                "duration_ms": int(duration or 0),
            }
        )
    return points
