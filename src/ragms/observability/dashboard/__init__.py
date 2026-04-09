"""Dashboard shell helpers."""

from .app import build_dashboard_context, render_app_shell
from .pages import render_data_browser, render_system_overview

__all__ = ["build_dashboard_context", "render_app_shell", "render_data_browser", "render_system_overview"]
