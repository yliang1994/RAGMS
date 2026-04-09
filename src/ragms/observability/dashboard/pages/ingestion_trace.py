"""Ingestion trace page renderer."""

from __future__ import annotations

from typing import Any

from ragms.observability.dashboard.components import (
    render_duration_chart,
    render_empty_state,
    render_table,
    render_trace_timeline,
)


def render_ingestion_trace(
    context: Any,
    *,
    status: str | None = None,
    collection: str | None = None,
    trace_id: str | None = None,
    renderer: Any | None = None,
) -> dict[str, Any]:
    """Render or return the ingestion trace browser payload."""

    filters = {
        key: value
        for key, value in {
            "status": status,
            "collection": collection,
            "trace_id": trace_id,
        }.items()
        if value not in (None, "")
    }
    all_traces = context.trace_service.list_traces(trace_type="ingestion")
    traces = context.trace_service.list_traces(
        trace_type="ingestion",
        status=status,
        collection=collection,
        trace_id=trace_id,
    )
    selected_trace_id = traces[0]["trace_id"] if traces else None
    selected_trace = (
        context.trace_service.get_trace_detail(selected_trace_id)
        if selected_trace_id is not None
        else None
    )
    timeline = render_trace_timeline(selected_trace)

    payload = {
        "kind": "ingestion_trace",
        "title": "Ingestion 追踪",
        "filters": filters,
        "filter_model": _build_filter_model(all_traces),
        "trace_list": render_table(
            traces,
            columns=[
                "trace_id",
                "status",
                "collection",
                "document_id",
                "source_path",
                "total_chunks",
                "total_images",
                "started_at",
                "target_page",
            ],
            empty_title="暂无 Ingestion Trace",
            empty_description="当前过滤条件下没有匹配的 ingestion trace。",
        ),
        "selected_trace_id": selected_trace_id,
        "selected_trace": selected_trace,
        "timeline": timeline,
        "timeline_table": render_table(
            timeline.get("timeline") or [],
            columns=[
                "stage_name",
                "raw_stage_name",
                "elapsed_ms",
                "provider",
                "progress",
                "input_summary",
                "output_summary",
                "metadata",
                "error",
            ],
            empty_title="暂无阶段时间线",
            empty_description="当前 trace 没有阶段详情。",
        ),
        "duration_chart": render_duration_chart(timeline.get("timeline") or []),
        "trace_empty_state": None
        if selected_trace is not None
        else render_empty_state("暂无 Trace 详情", "请选择一个 ingestion trace 查看阶段回放。"),
        "navigation": _build_navigation(selected_trace),
    }
    if renderer is not None:
        _render_ingestion_trace_streamlit(payload, renderer)
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


def _render_ingestion_trace_streamlit(payload: dict[str, Any], renderer: Any) -> None:
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
        return

    renderer.code(str(payload["selected_trace"]["summary"]), language="python")
    renderer.markdown("#### Timeline")
    _render_table_or_empty(payload["timeline_table"], renderer)
    if payload["timeline"]["omitted_stage_names"]:
        renderer.caption(f"Omitted stages: {', '.join(payload['timeline']['omitted_stage_names'])}")
    renderer.markdown("#### Duration Trend")
    if payload["duration_chart"]["point_count"] == 0:
        render_empty_state("暂无阶段耗时", "当前 trace 没有可展示的阶段耗时。", renderer=renderer)
    else:
        render_duration_chart(payload["duration_chart"]["points"], renderer=renderer)


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
