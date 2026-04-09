"""Evaluation panel page renderer."""

from __future__ import annotations

from typing import Any

from ragms.observability.dashboard.components import render_empty_state, render_metric_cards, render_table


def render_evaluation_panel(
    context: Any,
    *,
    run_id: str | None = None,
    renderer: Any | None = None,
) -> dict[str, Any]:
    """Render or return the evaluation panel payload."""

    runs = context.report_service.list_evaluation_runs()
    selected_run_id = run_id or (runs[0]["run_id"] if runs else None)
    selected_report = (
        context.report_service.load_report_detail(selected_run_id)
        if selected_run_id is not None
        else None
    )
    payload = {
        "kind": "evaluation_panel",
        "title": "评估面板",
        "reports": render_table(
            runs,
            columns=["run_id", "collection", "dataset_version", "quality_gate_status", "updated_at", "path"],
            empty_title="暂无评估报告",
            empty_description="当前没有本地报告，可先使用样例或等待阶段 H 的真实评估运行。",
        ),
        "selected_run_id": selected_run_id,
        "selected_report": selected_report,
        "metric_cards": render_metric_cards((selected_report or {}).get("metrics_summary") or {}),
        "report_empty_state": None
        if selected_report is not None
        else render_empty_state("暂无报告详情", "请选择一个评估报告后查看指标和配置摘要。"),
        "navigation": _build_navigation(selected_report),
    }
    if renderer is not None:
        _render_evaluation_panel_streamlit(payload, renderer)
    return payload


def _build_navigation(selected_report: dict[str, Any] | None) -> list[dict[str, Any]]:
    navigation = [{"label": "跳转到系统总览", "target_page": "system_overview"}]
    if selected_report is not None:
        navigation.extend(selected_report.get("navigation") or [])
    return navigation


def _render_evaluation_panel_streamlit(payload: dict[str, Any], renderer: Any) -> None:
    renderer.subheader(payload["title"])
    renderer.markdown("### Reports")
    _render_table_or_empty(payload["reports"], renderer)
    renderer.markdown("### Selected Report")
    if payload["selected_report"] is None:
        render_empty_state(
            payload["report_empty_state"]["title"],
            payload["report_empty_state"]["description"],
            renderer=renderer,
        )
        return
    render_metric_cards(payload["metric_cards"], renderer=renderer)
    renderer.code(str(payload["selected_report"]["report"]), language="python")


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
