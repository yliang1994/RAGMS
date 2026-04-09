"""Evaluation panel page renderer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ragms.observability.dashboard.components import render_empty_state, render_metric_cards, render_table
from ragms.core.trace_collector.trace_utils import serialize_exception


def render_evaluation_panel(
    context: Any,
    *,
    run_id: str | None = None,
    run_request: dict[str, Any] | None = None,
    compare_run_id: str | None = None,
    set_baseline_run_id: str | None = None,
    clear_baseline_scope: dict[str, Any] | None = None,
    renderer: Any | None = None,
) -> dict[str, Any]:
    """Render or return the evaluation panel payload."""

    run_form = render_evaluation_run_form(context, request=run_request)
    run_state = start_dashboard_evaluation(context, request=run_request)
    baseline_action = None
    if set_baseline_run_id is not None:
        baseline_action = context.report_service.set_baseline(set_baseline_run_id)
    elif clear_baseline_scope is not None:
        baseline_action = context.report_service.set_baseline(
            collection=clear_baseline_scope.get("collection"),
            dataset_version=clear_baseline_scope.get("dataset_version"),
            backend_set=list(clear_baseline_scope.get("backend_set") or []),
        )
    runs = context.report_service.list_evaluation_runs()
    selected_run_id = run_state.get("run_id") or run_id or (runs[0]["run_id"] if runs else None)
    selected_report = (
        context.report_service.load_report_detail(selected_run_id)
        if selected_run_id is not None
        else None
    )
    comparison = None
    if selected_run_id and compare_run_id:
        comparison = context.report_service.compare_runs(selected_run_id, compare_run_id)
    elif selected_run_id:
        comparison = context.report_service.compare_against_baseline(selected_run_id)
    baseline_actions = render_baseline_actions(context, selected_report, last_action=baseline_action)
    results = render_evaluation_results(
        selected_report,
        run_state=run_state,
        comparison=comparison,
        runs=runs,
    )
    payload = {
        "kind": "evaluation_panel",
        "title": "评估面板",
        "run_form": run_form,
        "run_state": run_state,
        "reports": render_table(
            runs,
            columns=["run_id", "collection", "dataset_version", "quality_gate_status", "is_baseline", "updated_at", "path"],
            empty_title="暂无评估报告",
            empty_description="当前没有本地报告，可先使用样例或等待阶段 H 的真实评估运行。",
        ),
        "selected_run_id": selected_run_id,
        "selected_report": selected_report,
        "metric_cards": results.get("metric_cards") or [],
        "report_empty_state": None
        if selected_report is not None
        else render_empty_state("暂无报告详情", "请选择一个评估报告后查看指标和配置摘要。"),
        "navigation": _build_navigation(selected_report),
        "baseline_actions": baseline_actions,
        "results": results,
    }
    if renderer is not None:
        _render_evaluation_panel_streamlit(payload, renderer)
    return payload


def render_evaluation_run_form(context: Any, *, request: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the evaluation run form payload from local datasets and settings."""

    dataset_root = context.settings.paths.data_dir / "evaluation" / "datasets"
    dataset_names = sorted(path.name for path in dataset_root.iterdir() if path.is_dir()) if dataset_root.is_dir() else []
    selected_dataset = str((request or {}).get("dataset_name") or (dataset_names[0] if dataset_names else ""))
    version_dir = dataset_root / selected_dataset
    versions = sorted(path.stem for path in version_dir.glob("*.json")) if version_dir.is_dir() else []
    return {
        "kind": "evaluation_run_form",
        "collections": [context.settings.vector_store.collection],
        "dataset_names": dataset_names,
        "dataset_versions": versions,
        "selected_collection": str((request or {}).get("collection") or context.settings.vector_store.collection),
        "selected_dataset_name": selected_dataset or None,
        "selected_dataset_version": str((request or {}).get("dataset_version") or (versions[-1] if versions else "")) or None,
        "selected_backend_set": list((request or {}).get("backend_set") or context.settings.evaluation.backends),
        "eval_options": {
            "top_k": int((request or {}).get("top_k") or 5),
            "labels": list((request or {}).get("labels") or []),
        },
    }


def start_dashboard_evaluation(context: Any, *, request: dict[str, Any] | None) -> dict[str, Any]:
    """Execute a dashboard-triggered evaluation run through the injected runner."""

    if request is None:
        return {"status": "idle"}
    if bool(request.get("simulate_running")):
        return {
            "status": "running",
            "message": "评估运行中，请稍后刷新报告列表。",
        }
    try:
        result = context.eval_runner.run(
            collection=request.get("collection"),
            dataset_name=str(request["dataset_name"]),
            dataset_version=request.get("dataset_version"),
            backend_set=list(request.get("backend_set") or []),
            labels=list(request.get("labels") or []),
            top_k=int(request.get("top_k") or 5),
        )
    except Exception as exc:
        return {
            "status": "failed",
            "error": serialize_exception(exc),
        }
    return {
        "status": "succeeded",
        "run_id": result.get("run_id"),
        "trace_id": result.get("trace_id"),
        "path": result.get("path"),
    }


def render_evaluation_results(
    selected_report: dict[str, Any] | None,
    *,
    run_state: dict[str, Any],
    comparison: dict[str, Any] | None,
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the result panel payload for running, success, failure, and empty states."""

    if run_state.get("status") == "running":
        return {
            "kind": "running",
            "message": run_state.get("message"),
            "metric_cards": [],
        }
    if run_state.get("status") == "failed":
        return {
            "kind": "failed",
            "error": run_state.get("error"),
            "metric_cards": [],
        }
    if selected_report is None:
        return {
            "kind": "empty",
            "metric_cards": [],
        }
    report = dict(selected_report.get("report") or {})
    scope = selected_report.get("baseline_scope")
    scope_runs = [row for row in runs if row.get("baseline_scope") == scope] if scope else []
    return {
        "kind": "succeeded",
        "metric_cards": render_metric_cards(selected_report.get("metrics_summary") or {}),
        "failed_samples": render_table(
            list(report.get("failed_samples") or []),
            columns=["sample_id", "stage", "error"],
            empty_title="暂无失败样本",
            empty_description="当前报告没有失败样本。",
        ),
        "config_snapshot": dict(report.get("config_snapshot") or {}),
        "comparison": comparison,
        "trend": scope_runs,
        "provider_compare": [
            {
                "run_id": row.get("run_id"),
                "backend_set": ",".join(row.get("backend_set") or []),
                **dict(row.get("metrics_summary") or {}),
            }
            for row in scope_runs
        ],
    }


def render_baseline_actions(
    context: Any,
    selected_report: dict[str, Any] | None,
    *,
    last_action: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return current baseline state and action affordances for the selected run."""

    if selected_report is None:
        return {
            "kind": "baseline_actions",
            "current_baseline": None,
            "available": False,
            "last_action": last_action,
        }
    backend_set = list(selected_report.get("backend_set") or [])
    dataset_version = str(selected_report.get("dataset_version") or "").strip()
    if not backend_set or not dataset_version:
        return {
            "kind": "baseline_actions",
            "current_baseline": None,
            "available": False,
            "selected_run_id": selected_report.get("run_id"),
            "can_set_baseline": False,
            "can_clear_baseline": False,
            "last_action": last_action,
        }
    current_baseline = context.report_service.get_baseline(
        collection=str(selected_report.get("collection")),
        dataset_version=dataset_version,
        backend_set=backend_set,
    )
    return {
        "kind": "baseline_actions",
        "current_baseline": current_baseline,
        "available": True,
        "selected_run_id": selected_report.get("run_id"),
        "can_set_baseline": current_baseline is None or current_baseline.get("run_id") != selected_report.get("run_id"),
        "can_clear_baseline": current_baseline is not None,
        "last_action": last_action,
    }


def _build_navigation(selected_report: dict[str, Any] | None) -> list[dict[str, Any]]:
    navigation = [{"label": "跳转到系统总览", "target_page": "system_overview"}]
    if selected_report is not None:
        navigation.extend(selected_report.get("navigation") or [])
    return navigation


def _render_evaluation_panel_streamlit(payload: dict[str, Any], renderer: Any) -> None:
    renderer.subheader(payload["title"])
    renderer.code(str(payload["run_form"]), language="python")
    if payload["run_state"]["status"] != "idle":
        renderer.code(str(payload["run_state"]), language="python")
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
    renderer.code(str(payload["baseline_actions"]), language="python")
    renderer.code(str(payload["results"]), language="python")
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
