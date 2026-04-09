from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from ragms.observability.dashboard.components import (
    render_duration_chart,
    render_empty_state,
    render_metric_cards,
    render_status_badge,
    render_table,
    render_trace_timeline,
)
from tests.integration.test_dashboard_data_access import _seed_metadata


def _write_settings(path: Path) -> Path:
    path.write_text(
        textwrap.dedent(
            """
            app_name: ragms
            environment: test
            paths:
              project_root: .
              data_dir: data
              logs_dir: logs
            llm:
              provider: openai
              model: gpt-4.1-mini
            embedding:
              provider: openai
              model: text-embedding-3-small
            vector_store:
              backend: chroma
              collection: dashboard-demo
            retrieval:
              strategy: hybrid
              fusion_algorithm: rrf
              rerank_backend: disabled
            evaluation:
              backends: [custom_metrics]
            observability:
              enabled: true
              log_file: logs/traces.jsonl
              log_level: INFO
            dashboard:
              enabled: true
              port: 8501
              traces_file: logs/traces.jsonl
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.integration
def test_dashboard_components_render_empty_table_badges_and_metric_cards(tmp_path: Path) -> None:
    empty_state = render_empty_state("No traces", "Nothing has been recorded yet.")
    table = render_table([], empty_title="No documents", empty_description="Seed data is missing.")
    badge = render_status_badge("partial_success")
    cards = render_metric_cards({"documents": 2, "chunks": 4})

    assert empty_state["title"] == "No traces"
    assert table["kind"] == "empty"
    assert table["empty_state"]["title"] == "No documents"
    assert badge["label"] == "Partial Success"
    assert cards[0]["label"] == "documents"


@pytest.mark.integration
def test_dashboard_trace_components_render_comparison_and_timeline(tmp_path: Path) -> None:
    settings_path = _write_settings(tmp_path / "settings.yaml")
    _data_service, trace_service, _report_service = _seed_metadata(settings_path)

    comparison = trace_service.compare_traces("trace-left", "trace-right")
    left_trace = trace_service.get_trace_detail("trace-left")
    chart = render_duration_chart(comparison)
    timeline = render_trace_timeline(left_trace)

    assert chart["point_count"] >= 2
    assert chart["points"][1]["label"] == "dense_retrieval"
    assert timeline["trace_type"] == "query"
    assert timeline["timeline"][0]["stage_name"] == "query_processing"
    assert timeline["timeline"][1]["metadata"]["provider"] == "chroma"
