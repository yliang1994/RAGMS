"""Dashboard shell helpers."""

from .app import build_dashboard_context, render_app_shell, resolve_dashboard_navigation_target
from .pages import (
    render_data_browser,
    render_evaluation_panel,
    render_ingestion_management,
    render_ingestion_trace,
    render_query_trace,
    render_system_overview,
)

__all__ = [
    "build_dashboard_context",
    "render_app_shell",
    "resolve_dashboard_navigation_target",
    "render_data_browser",
    "render_evaluation_panel",
    "render_ingestion_management",
    "render_ingestion_trace",
    "render_query_trace",
    "render_system_overview",
]
