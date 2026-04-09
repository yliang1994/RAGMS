"""Trace timeline rendering helpers for ingestion, query, and evaluation traces."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .tables import render_empty_state, render_status_badge

INGESTION_STAGE_ORDER = [
    "file_integrity",
    "load",
    "chunking",
    "transform",
    "embedding",
    "storage",
    "lifecycle_finalize",
]
_INGESTION_STAGE_ALIASES = {
    "split": "chunking",
    "embed": "embedding",
    "upsert": "storage",
}


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
            "omitted_stage_names": [],
            "progress_summary": {"completed": 0, "expected": 0},
            "empty_state": render_empty_state(
                "No trace stages",
                "The selected trace does not contain any stage records.",
                renderer=renderer,
            ),
        }

    ordered_stages = _ordered_stages(trace)
    timeline = [_build_timeline_item(trace, stage) for stage in ordered_stages]
    omitted_stage_names = _omitted_stage_names(trace, ordered_stages)
    payload = {
        "trace_id": trace.get("trace_id"),
        "trace_type": trace.get("trace_type"),
        "status": trace.get("status"),
        "timeline": timeline,
        "omitted_stage_names": omitted_stage_names,
        "progress_summary": {
            "completed": len(timeline),
            "expected": len(INGESTION_STAGE_ORDER)
            if str(trace.get("trace_type") or "").strip().lower() == "ingestion"
            else len(timeline),
        },
    }
    if renderer is not None:
        for item in timeline:
            renderer.write(
                f"{item['stage_name']} | {item['status_badge']['label']} | {item['elapsed_ms']} ms"
            )
        if omitted_stage_names:
            renderer.caption(f"Omitted stages: {', '.join(omitted_stage_names)}")
    return payload


def _ordered_stages(trace: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    stages = list(trace.get("stages") or [])
    if str(trace.get("trace_type") or "").strip().lower() != "ingestion":
        return stages

    order_map = {name: index for index, name in enumerate(INGESTION_STAGE_ORDER)}
    return sorted(
        stages,
        key=lambda stage: (
            order_map.get(_normalize_ingestion_stage_name(stage.get("stage_name")), len(order_map)),
            order_map.get(str(stage.get("stage_name") or "").strip(), len(order_map)),
        ),
    )


def _build_timeline_item(trace: Mapping[str, Any], stage: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(stage.get("metadata") or {})
    canonical_stage_name = _canonical_stage_name(trace, stage.get("stage_name"))
    return {
        "stage_name": canonical_stage_name,
        "raw_stage_name": stage.get("stage_name"),
        "status_badge": render_status_badge(stage.get("status")),
        "elapsed_ms": int(stage.get("elapsed_ms") or 0),
        "input_summary": stage.get("input_summary"),
        "output_summary": stage.get("output_summary"),
        "metadata": metadata,
        "provider": metadata.get("provider"),
        "progress": _stage_progress(stage, metadata),
        "error": stage.get("error"),
    }


def _canonical_stage_name(trace: Mapping[str, Any], stage_name: Any) -> str:
    normalized = str(stage_name or "").strip()
    if str(trace.get("trace_type") or "").strip().lower() != "ingestion":
        return normalized
    return _normalize_ingestion_stage_name(normalized)


def _normalize_ingestion_stage_name(stage_name: Any) -> str:
    normalized = str(stage_name or "").strip()
    return _INGESTION_STAGE_ALIASES.get(normalized, normalized)


def _omitted_stage_names(trace: Mapping[str, Any], stages: list[Mapping[str, Any]]) -> list[str]:
    if str(trace.get("trace_type") or "").strip().lower() != "ingestion":
        return []

    present = {
        _normalize_ingestion_stage_name(stage.get("stage_name"))
        for stage in stages
    }
    return [stage_name for stage_name in INGESTION_STAGE_ORDER if stage_name not in present]


def _stage_progress(stage: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        "retry_count": metadata.get("retry_count", 0),
        "fallback_reason": metadata.get("fallback_reason"),
    }
    for key in ("retrieved_count", "chunk_count", "upsert_count", "batch_count"):
        if key in metadata:
            summary[key] = metadata.get(key)
    if stage.get("status") == "skipped":
        summary["skipped"] = True
    return summary
