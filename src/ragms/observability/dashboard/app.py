"""Streamlit dashboard shell and shared context bootstrap."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from collections.abc import Sequence
from typing import Any

from ragms.core.evaluation import ReportService
from ragms.core.management import DataService, TraceService
from ragms.observability.dashboard.pages import render_data_browser, render_system_overview
from ragms.runtime.container import PlaceholderService, ServiceContainer, build_container
from ragms.runtime.config import load_settings
from ragms.runtime.settings_models import AppSettings


@dataclass(frozen=True)
class DashboardPage:
    """One registered dashboard page entry."""

    key: str
    title: str
    description: str
    status: str = "placeholder"


@dataclass
class DashboardContext:
    """Shared shell context injected into all dashboard pages."""

    settings: AppSettings
    runtime: ServiceContainer
    data_service: DataService
    trace_service: TraceService
    document_admin_service: Any
    report_service: Any
    pages: list[DashboardPage]
    service_snapshot: dict[str, Any]


PAGE_REGISTRY = [
    DashboardPage(
        key="system_overview",
        title="系统总览",
        description="系统指标、最近 trace 和组件配置摘要。",
        status="ready",
    ),
    DashboardPage(
        key="data_browser",
        title="数据浏览器",
        description="集合、文档、chunk 与元数据浏览。",
        status="ready",
    ),
    DashboardPage(
        key="ingestion_management",
        title="Ingestion管理",
        description="摄取任务和文档状态管理占位页。",
    ),
    DashboardPage(
        key="ingestion_trace",
        title="Ingestion追踪",
        description="按 trace_id 查看摄取链路详情。",
    ),
    DashboardPage(
        key="query_trace",
        title="Query追踪",
        description="按 trace_id 查看查询链路详情。",
    ),
    DashboardPage(
        key="evaluation_panel",
        title="评估面板",
        description="评估结果和报告读取占位页。",
    ),
]


def build_dashboard_context(
    settings: AppSettings,
    *,
    runtime: ServiceContainer | None = None,
    data_service: DataService | None = None,
    trace_service: TraceService | None = None,
    document_admin_service: Any | None = None,
    report_service: Any | None = None,
) -> DashboardContext:
    """Build the shared dashboard context from the runtime and management services."""

    resolved_runtime = runtime or build_container(settings)
    resolved_data_service = data_service or DataService(settings)
    resolved_trace_service = trace_service or TraceService(settings)
    resolved_document_admin = document_admin_service or resolved_runtime.services.get(
        "document_admin_service",
        PlaceholderService(name="document_admin_service", implementation="pending", config={}),
    )
    resolved_report_service = report_service or ReportService(settings)
    service_snapshot = {
        "data_service": resolved_data_service.__class__.__name__,
        "trace_service": resolved_trace_service.__class__.__name__,
        "document_admin_service": getattr(resolved_document_admin, "name", resolved_document_admin.__class__.__name__),
        "report_service": resolved_report_service.__class__.__name__,
    }
    return DashboardContext(
        settings=settings,
        runtime=resolved_runtime,
        data_service=resolved_data_service,
        trace_service=resolved_trace_service,
        document_admin_service=resolved_document_admin,
        report_service=resolved_report_service,
        pages=list(PAGE_REGISTRY),
        service_snapshot=service_snapshot,
    )


def render_app_shell(
    context: DashboardContext,
    *,
    selected_page: str | None = None,
    renderer: Any | None = None,
) -> dict[str, Any]:
    """Render the dashboard shell or return a serializable shell snapshot for tests."""

    page_keys = [page.key for page in context.pages]
    active_page = selected_page if selected_page in page_keys else page_keys[0]
    page = next(page for page in context.pages if page.key == active_page)
    shell_payload = {
        "title": context.settings.dashboard.title,
        "selected_page": active_page,
        "pages": [asdict(item) for item in context.pages],
        "auto_refresh": context.settings.dashboard.auto_refresh,
        "refresh_interval": context.settings.dashboard.refresh_interval,
        "port": context.settings.dashboard.port,
        "traces_file": str(context.settings.dashboard.traces_file),
        "service_snapshot": dict(context.service_snapshot),
        "placeholder": {
            "title": page.title,
            "description": page.description,
            "status": page.status,
        },
        "page": _render_page_payload(context, active_page),
    }
    if renderer is None:
        return shell_payload

    renderer.set_page_config(page_title=context.settings.dashboard.title, layout="wide")
    renderer.title(context.settings.dashboard.title)
    renderer.caption(
        f"port={context.settings.dashboard.port} "
        f"auto_refresh={context.settings.dashboard.auto_refresh} "
        f"refresh_interval={context.settings.dashboard.refresh_interval}s"
    )
    renderer.code(str(context.service_snapshot), language="python")
    renderer.sidebar.title("导航")
    chosen_page = renderer.sidebar.radio(
        "页面",
        options=page_keys,
        index=page_keys.index(active_page),
        format_func=lambda key: next(item.title for item in context.pages if item.key == key),
    )
    active_page = chosen_page
    page = next(item for item in context.pages if item.key == active_page)
    shell_payload["selected_page"] = active_page
    shell_payload["placeholder"] = {
        "title": page.title,
        "description": page.description,
        "status": page.status,
    }
    shell_payload["page"] = _render_page_payload(context, active_page)
    _render_page(context, active_page, renderer)
    return shell_payload


def _render_page_payload(context: DashboardContext, active_page: str) -> dict[str, Any]:
    if active_page == "system_overview":
        return render_system_overview(context)
    if active_page == "data_browser":
        return render_data_browser(context)
    return {
        "kind": "placeholder",
        "title": next(page.title for page in context.pages if page.key == active_page),
        "description": next(page.description for page in context.pages if page.key == active_page),
        "status": next(page.status for page in context.pages if page.key == active_page),
    }


def _render_page(context: DashboardContext, active_page: str, renderer: Any) -> None:
    if active_page == "system_overview":
        render_system_overview(context, renderer=renderer)
        return
    if active_page == "data_browser":
        render_data_browser(context, renderer=renderer)
        return
    placeholder = _render_page_payload(context, active_page)
    renderer.subheader(placeholder["title"])
    renderer.info(placeholder["description"])


def main(argv: Sequence[str] | None = None) -> int:
    """Render the dashboard shell inside a Streamlit execution context."""

    parser = argparse.ArgumentParser(description="Render the RagMS dashboard shell.")
    parser.add_argument("--settings", default="settings.yaml")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        import streamlit as st
    except Exception as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("streamlit is required to render the dashboard shell") from exc

    settings = load_settings(args.settings)
    context = build_dashboard_context(settings)
    render_app_shell(context, renderer=st)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
