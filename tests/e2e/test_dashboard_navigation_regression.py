from __future__ import annotations

from pathlib import Path

import pytest

from ragms.observability.dashboard import build_dashboard_context, resolve_dashboard_navigation_target
from ragms.runtime.config import load_settings
from tests.e2e.test_dashboard_smoke import _write_settings
from tests.integration.test_dashboard_data_access import _seed_metadata
from tests.integration.test_dashboard_ingestion_trace import _append_ingestion_variants
from tests.integration.test_dashboard_query_trace import _append_query_variants


def assert_dashboard_navigation_regression(tmp_path: Path) -> dict[str, object]:
    """Assert final dashboard navigation targets remain stable."""

    settings_path = _write_settings(tmp_path / "settings.yaml")
    data_service, trace_service, report_service = _append_query_variants(settings_path)
    settings = load_settings(settings_path)
    context = build_dashboard_context(
        settings,
        data_service=data_service,
        trace_service=trace_service,
        report_service=report_service,
    )
    browser = resolve_dashboard_navigation_target(
        context,
        {"target_page": "data_browser", "document_id": "doc-1"},
    )
    query_compare = resolve_dashboard_navigation_target(
        context,
        {
            "target_page": "query_trace",
            "left_trace_id": "trace-query-full",
            "right_trace_id": "trace-query-retrieval-only",
        },
    )
    evaluation = resolve_dashboard_navigation_target(
        context,
        {"target_page": "evaluation_panel", "run_id": "run-1"},
    )
    return {
        "browser": browser,
        "query_compare": query_compare,
        "evaluation": evaluation,
    }


@pytest.mark.e2e
def test_dashboard_navigation_regression_covers_cross_page_targets(tmp_path: Path) -> None:
    resolved = assert_dashboard_navigation_regression(tmp_path)

    assert resolved["browser"]["kind"] == "data_browser"
    assert resolved["browser"]["selected_document"]["document_id"] == "doc-1"
    assert resolved["query_compare"]["kind"] == "query_trace"
    assert resolved["query_compare"]["comparison_selection"]["left_trace_id"] == "trace-query-full"
    assert resolved["evaluation"]["kind"] == "evaluation_panel"
    assert resolved["evaluation"]["selected_report"]["run_id"] == "run-1"
