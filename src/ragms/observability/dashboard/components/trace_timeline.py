"""Trace timeline rendering helpers for ingestion, query, and evaluation traces."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .tables import render_empty_state, render_status_badge


def render_trace_timeline(
    trace: Mapping[str, Any] | None,
    *,
    renderer: Any | None = None,
) -> dict[str, Any]:
    """Render or return a stable timeline payload from a trace detail."""

    if not trace or not trace.get("stages"):
        return {
            "trace_id": None if trace is None else trace.get("trace_id"),
            "trace_type": None if trace is None else trace.get("trace_type"),
            "timeline": [],
            "empty_state": render_empty_state(
                "No trace stages",
                "The selected trace does not contain any stage records.",
                renderer=renderer,
            ),
        }

    timeline = [
        {
            "stage_name": stage.get("stage_name"),
            "status_badge": render_status_badge(stage.get("status")),
            "elapsed_ms": int(stage.get("elapsed_ms") or 0),
            "input_summary": stage.get("input_summary"),
            "output_summary": stage.get("output_summary"),
            "metadata": dict(stage.get("metadata") or {}),
            "error": stage.get("error"),
        }
        for stage in trace.get("stages") or []
    ]
    payload = {
        "trace_id": trace.get("trace_id"),
        "trace_type": trace.get("trace_type"),
        "status": trace.get("status"),
        "timeline": timeline,
    }
    if renderer is not None:
        for item in timeline:
            renderer.write(
                f"{item['stage_name']} | {item['status_badge']['label']} | {item['elapsed_ms']} ms"
            )
    return payload
