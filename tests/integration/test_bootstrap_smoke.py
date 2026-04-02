from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from scripts.ingest_documents import ingest_documents_main
from scripts.query_cli import run_cli
from scripts.run_dashboard import run_dashboard
from scripts.run_mcp_server import run_mcp_server_main


def write_settings(path: Path) -> Path:
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
              collection: smoke-tests
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
def test_bootstrap_entrypoints_start_with_local_settings(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings_path = write_settings(tmp_path / "settings.yaml")

    assert run_cli(["--settings", str(settings_path), "what is ragms"]) == 0
    assert run_dashboard(["--settings", str(settings_path)]) == 0
    assert run_mcp_server_main(["--settings", str(settings_path)]) == 0
    assert ingest_documents_main(
        ["--settings", str(settings_path), "--source-dir", "fixtures/documents"]
    ) == 0

    output = capsys.readouterr().out
    assert "Query CLI ready" in output
    assert "Dashboard bootstrap ready" in output
    assert "MCP server bootstrap ready" in output
    assert "Ingestion bootstrap ready" in output
