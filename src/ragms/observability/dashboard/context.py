"""Dashboard shared context bootstrap and service wiring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ragms.core.evaluation import EvalRunner, ReportService
from ragms.core.management import DataService, DocumentAdminService, TraceService
from ragms.core.query_engine import build_query_engine
from ragms.runtime.container import ServiceContainer, build_container
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
    eval_runner: Any
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
        status="ready",
    ),
    DashboardPage(
        key="query_trace",
        title="Query追踪",
        description="按 trace_id 查看查询链路详情。",
        status="ready",
    ),
    DashboardPage(
        key="evaluation_panel",
        title="评估面板",
        description="评估结果和报告读取占位页。",
        status="ready",
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
    eval_runner: Any | None = None,
) -> DashboardContext:
    """Build the shared dashboard context from runtime and management services."""

    resolved_runtime = runtime or build_container(settings)
    resolved_data_service = data_service or DataService(settings)
    resolved_trace_service = trace_service or TraceService(settings)
    resolved_document_admin = document_admin_service or DocumentAdminService(settings)
    resolved_report_service = report_service or ReportService(settings)
    resolved_eval_runner = eval_runner or EvalRunner(
        settings=settings,
        report_service=resolved_report_service,
        query_engine=build_query_engine(resolved_runtime, settings=settings),
    )
    service_snapshot = {
        "data_service": resolved_data_service.__class__.__name__,
        "trace_service": resolved_trace_service.__class__.__name__,
        "document_admin_service": getattr(resolved_document_admin, "name", resolved_document_admin.__class__.__name__),
        "report_service": resolved_report_service.__class__.__name__,
        "eval_runner": resolved_eval_runner.__class__.__name__,
    }
    return DashboardContext(
        settings=settings,
        runtime=resolved_runtime,
        data_service=resolved_data_service,
        trace_service=resolved_trace_service,
        document_admin_service=resolved_document_admin,
        report_service=resolved_report_service,
        eval_runner=resolved_eval_runner,
        pages=list(PAGE_REGISTRY),
        service_snapshot=service_snapshot,
    )
