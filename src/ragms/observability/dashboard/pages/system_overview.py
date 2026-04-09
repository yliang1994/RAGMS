"""System overview page renderer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ragms.observability.dashboard.components import (
    render_duration_chart,
    render_empty_state,
    render_metric_cards,
    render_table,
)


def render_system_overview(
    context: Any,
    *,
    renderer: Any | None = None,
) -> dict[str, Any]:
    """Render or return the system overview page payload."""

    overview = context.data_service.get_system_overview_metrics()
    collection_statistics = context.data_service.get_collection_statistics()
    recent_traces = _decorate_traces(context.trace_service.list_traces(limit=8))
    recent_ingestion = _decorate_traces(context.trace_service.list_traces(trace_type="ingestion", limit=5))
    recent_queries = _decorate_traces(context.trace_service.list_traces(trace_type="query", limit=5))
    recent_failures = _decorate_traces(context.trace_service.get_recent_failures(limit=5))
    duration_trend = render_duration_chart(
        [
            {
                "trace_id": trace.get("trace_id"),
                "duration_ms": trace.get("duration_ms"),
            }
            for trace in recent_traces
            if trace.get("duration_ms") is not None
        ]
    )
    duration_trend_empty_state = None
    if duration_trend["point_count"] == 0:
        duration_trend_empty_state = render_empty_state(
            "暂无耗时趋势",
            "最近 trace 尚未产生可展示的耗时数据。",
        )

    page_payload = {
        "kind": "system_overview",
        "title": "系统总览",
        "metric_cards": render_metric_cards(
            [
                {"label": "Collections", "value": overview["collection_count"]},
                {"label": "Documents", "value": overview["document_count"]},
                {"label": "Chunks", "value": overview["chunk_count"]},
                {"label": "Images", "value": overview["image_count"]},
            ]
        ),
        "status_counts": dict(overview.get("status_counts") or {}),
        "collection_statistics": render_table(
            _decorate_collection_rows(collection_statistics.get("collections") or []),
            columns=["name", "document_count", "chunk_count", "image_count", "latest_updated_at", "target_page"],
            empty_title="暂无集合数据",
            empty_description="本地索引和元数据尚未生成，可先执行文档摄取。",
        ),
        "recent_traces": render_table(
            recent_traces,
            columns=["trace_id", "trace_type", "status", "duration_ms", "started_at", "target_page"],
            empty_title="暂无 Trace",
            empty_description="当前还没有可展示的执行链路。",
        ),
        "recent_ingestion_traces": render_table(
            recent_ingestion,
            columns=["trace_id", "status", "duration_ms", "started_at", "source_path", "target_page"],
            empty_title="暂无 Ingestion 记录",
            empty_description="当前还没有摄取链路记录。",
        ),
        "recent_query_traces": render_table(
            recent_queries,
            columns=["trace_id", "status", "duration_ms", "started_at", "query", "target_page"],
            empty_title="暂无 Query 记录",
            empty_description="当前还没有查询链路记录。",
        ),
        "recent_failures": render_table(
            recent_failures,
            columns=["trace_id", "trace_type", "status", "started_at", "error", "target_page"],
            empty_title="暂无失败记录",
            empty_description="最近没有失败或部分成功的 trace。",
        ),
        "failure_summary": {
            "failure_count": len(recent_failures),
            "by_trace_type": _count_by_field(recent_failures, "trace_type"),
            "by_status": _count_by_field(recent_failures, "status"),
        },
        "duration_trend": duration_trend,
        "duration_trend_empty_state": duration_trend_empty_state,
        "config_summary": {
            **dict(overview.get("config_summary") or {}),
            "service_snapshot": dict(getattr(context, "service_snapshot", {}) or {}),
        },
        "navigation": [
            {"label": "查看数据浏览", "target_page": "data_browser"},
            {"label": "查看 Ingestion 追踪", "target_page": "ingestion_trace"},
            {"label": "查看 Query 追踪", "target_page": "query_trace"},
            {"label": "查看评估面板", "target_page": "evaluation_panel"},
        ],
    }
    if renderer is not None:
        _render_system_overview_streamlit(page_payload, renderer)
    return page_payload


def _decorate_traces(traces: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **dict(trace),
            "target_page": _trace_target_page(trace.get("trace_type")),
        }
        for trace in traces
    ]


def _decorate_collection_rows(collections: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **dict(collection),
            "target_page": "data_browser",
        }
        for collection in collections
    ]


def _count_by_field(rows: list[Mapping[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get(field) or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def _trace_target_page(trace_type: Any) -> str:
    if str(trace_type or "").strip().lower() == "ingestion":
        return "ingestion_trace"
    return "query_trace"


def _render_system_overview_streamlit(payload: dict[str, Any], renderer: Any) -> None:
    renderer.subheader(payload["title"])
    render_metric_cards(payload["metric_cards"], renderer=renderer)

    renderer.markdown("### 集合统计")
    render_table(payload["collection_statistics"]["rows"], columns=payload["collection_statistics"]["columns"], renderer=renderer)

    renderer.markdown("### 最近 Trace")
    _render_table_or_empty(payload["recent_traces"], renderer)

    renderer.markdown("### 最近失败")
    _render_table_or_empty(payload["recent_failures"], renderer)

    renderer.markdown("### 耗时趋势")
    if payload["duration_trend"]["point_count"] == 0:
        render_empty_state(
            payload["duration_trend_empty_state"]["title"],
            payload["duration_trend_empty_state"]["description"],
            renderer=renderer,
        )
    else:
        render_duration_chart(payload["duration_trend"]["points"], renderer=renderer)

    renderer.markdown("### 配置摘要")
    renderer.code(str(payload["config_summary"]), language="python")


def _render_table_or_empty(table_payload: dict[str, Any], renderer: Any) -> None:
    if table_payload["kind"] == "empty":
        render_empty_state(
            table_payload["empty_state"]["title"],
            table_payload["empty_state"]["description"],
            renderer=renderer,
        )
        return
    render_table(
        table_payload["rows"],
        columns=table_payload["columns"],
        renderer=renderer,
    )
