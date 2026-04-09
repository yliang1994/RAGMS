"""Query trace page renderer."""

from __future__ import annotations

from typing import Any

from ragms.observability.dashboard.components import (
    render_empty_state,
    render_query_trace_comparison,
    render_table,
    render_trace_timeline,
)


def render_query_trace(
    context: Any,
    *,
    status: str | None = None,
    collection: str | None = None,
    trace_id: str | None = None,
    left_trace_id: str | None = None,
    right_trace_id: str | None = None,
    renderer: Any | None = None,
) -> dict[str, Any]:
    """Render or return the query trace browser payload."""

    filters = {
        key: value
        for key, value in {
            "status": status,
            "collection": collection,
            "trace_id": trace_id,
        }.items()
        if value not in (None, "")
    }
    all_traces = context.trace_service.list_traces(trace_type="query")
    traces = context.trace_service.list_traces(
        trace_type="query",
        status=status,
        collection=collection,
        trace_id=trace_id,
    )
    selected_trace_id = left_trace_id or trace_id or (traces[0]["trace_id"] if traces else None)
    selected_trace = (
        context.trace_service.get_trace_detail(selected_trace_id)
        if selected_trace_id is not None
        else None
    )
    timeline = render_trace_timeline(selected_trace)

    resolved_left = left_trace_id or (traces[0]["trace_id"] if len(traces) >= 1 else None)
    resolved_right = right_trace_id or (traces[1]["trace_id"] if len(traces) >= 2 else None)
    comparison = (
        context.trace_service.compare_traces(resolved_left, resolved_right)
        if resolved_left is not None and resolved_right is not None
        else None
    )
    comparison_payload = (
        render_query_trace_comparison(comparison)
        if comparison is not None
        else {
            "kind": "empty",
            "empty_state": render_empty_state("暂无 Trace 对比", "至少需要两个 query trace 才能执行对比。"),
        }
    )

    payload = {
        "kind": "query_trace",
        "title": "Query 追踪",
        "filters": filters,
        "filter_model": _build_filter_model(all_traces),
        "trace_list": render_table(
            traces,
            columns=[
                "trace_id",
                "status",
                "collection",
                "query",
                "stage_count",
                "last_stage",
                "duration_ms",
                "started_at",
                "target_page",
            ],
            empty_title="暂无 Query Trace",
            empty_description="当前过滤条件下没有匹配的 query trace。",
        ),
        "selected_trace_id": selected_trace_id,
        "selected_trace": selected_trace,
        "timeline": timeline,
        "timeline_table": render_table(
            timeline.get("timeline") or [],
            columns=[
                "stage_name",
                "elapsed_ms",
                "provider",
                "progress",
                "input_summary",
                "output_summary",
                "metadata",
                "error",
            ],
            empty_title="暂无阶段时间线",
            empty_description="当前 query trace 没有阶段详情。",
        ),
        "comparison": comparison_payload,
        "comparison_selection": {
            "left_trace_id": resolved_left,
            "right_trace_id": resolved_right,
        },
        "trace_empty_state": None
        if selected_trace is not None
        else render_empty_state("暂无 Trace 详情", "请选择一个 query trace 查看阶段回放。"),
        "navigation": _build_navigation(selected_trace),
    }
    if renderer is not None:
        _render_query_trace_streamlit(payload, renderer)
    return payload


def _build_filter_model(traces: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "statuses": sorted(
            {
                str(trace.get("status") or "").strip()
                for trace in traces
                if str(trace.get("status") or "").strip()
            }
        ),
        "collections": sorted(
            {
                str(trace.get("collection") or "").strip()
                for trace in traces
                if str(trace.get("collection") or "").strip()
            }
        ),
        "trace_ids": [trace.get("trace_id") for trace in traces[:10]],
    }


def _build_navigation(selected_trace: dict[str, Any] | None) -> list[dict[str, Any]]:
    navigation = [{"label": "跳转到系统总览", "target_page": "system_overview"}]
    if selected_trace is None:
        return navigation
    navigation.extend(selected_trace.get("navigation") or [])
    return navigation


def _render_query_trace_streamlit(payload: dict[str, Any], renderer: Any) -> None:
    renderer.subheader(payload["title"])
    renderer.caption(f"当前过滤条件: {payload['filters'] or '无'}")

    renderer.markdown("### Filters")
    renderer.code(str(payload["filter_model"]), language="python")

    renderer.markdown("### Trace List")
    _render_table_or_empty(payload["trace_list"], renderer)

    renderer.markdown("### Trace Detail")
    if payload["selected_trace"] is None:
        render_empty_state(
            payload["trace_empty_state"]["title"],
            payload["trace_empty_state"]["description"],
            renderer=renderer,
        )
    else:
        renderer.code(str(payload["selected_trace"]["summary"]), language="python")
        renderer.markdown("#### Timeline")
        _render_table_or_empty(payload["timeline_table"], renderer)

    renderer.markdown("### Trace Comparison")
    if payload["comparison"].get("kind") == "empty":
        render_empty_state(
            payload["comparison"]["empty_state"]["title"],
            payload["comparison"]["empty_state"]["description"],
            renderer=renderer,
        )
    else:
        renderer.code(str(payload["comparison"]["metric_deltas"]), language="python")
        renderer.code(str(payload["comparison"]["query_differences"]), language="python")


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
