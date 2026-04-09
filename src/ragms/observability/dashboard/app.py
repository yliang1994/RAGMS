"""Streamlit dashboard shell and shared context bootstrap."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from collections.abc import Sequence
from typing import Any

from ragms.observability.dashboard.pages import (
    render_data_browser,
    render_evaluation_panel,
    render_ingestion_management,
    render_ingestion_trace,
    render_query_trace,
    render_system_overview,
)
from ragms.observability.dashboard.context import DashboardContext, build_dashboard_context
from ragms.runtime.config import load_settings


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
    if active_page == "ingestion_management":
        return render_ingestion_management(context)
    if active_page == "ingestion_trace":
        return render_ingestion_trace(context)
    if active_page == "query_trace":
        return render_query_trace(context)
    if active_page == "evaluation_panel":
        return render_evaluation_panel(context)
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
    if active_page == "ingestion_management":
        render_ingestion_management(context, renderer=renderer)
        return
    if active_page == "ingestion_trace":
        render_ingestion_trace(context, renderer=renderer)
        return
    if active_page == "query_trace":
        render_query_trace(context, renderer=renderer)
        return
    if active_page == "evaluation_panel":
        render_evaluation_panel(context, renderer=renderer)
        return
    placeholder = _render_page_payload(context, active_page)
    renderer.subheader(placeholder["title"])
    renderer.info(placeholder["description"])


def resolve_dashboard_navigation_target(
    context: DashboardContext,
    target: dict[str, Any],
) -> dict[str, Any]:
    """Resolve one dashboard navigation payload into the destination page snapshot."""

    target_page = str(target.get("target_page") or "").strip()
    if target_page == "data_browser":
        return render_data_browser(
            context,
            collection=target.get("collection"),
            document_id=target.get("document_id"),
            chunk_id=target.get("chunk_id"),
        )
    if target_page == "ingestion_trace":
        trace_ids = target.get("trace_ids") or []
        return render_ingestion_trace(
            context,
            trace_id=trace_ids[0] if trace_ids else target.get("trace_id"),
        )
    if target_page == "query_trace":
        return render_query_trace(
            context,
            trace_id=target.get("trace_id"),
            left_trace_id=target.get("left_trace_id"),
            right_trace_id=target.get("right_trace_id"),
        )
    if target_page == "evaluation_panel":
        return render_evaluation_panel(
            context,
            run_id=target.get("run_id"),
        )
    return _render_page_payload(context, target_page or "system_overview")


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
