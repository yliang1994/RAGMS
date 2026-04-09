"""One-command end-to-end acceptance runner with structured summary output."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ragms.core.management.trace_service import TraceService
from ragms.core.query_engine import build_query_engine
from ragms.mcp_server.tools.ingest import handle_ingest_documents
from ragms.runtime.container import ServiceContainer
from ragms.runtime.settings_models import AppSettings
from tests.e2e.test_dashboard_navigation_regression import assert_dashboard_navigation_regression
from tests.e2e.test_dashboard_smoke import dashboard_smoke_check
from tests.e2e.test_evaluation_visible_in_dashboard import assert_dashboard_regression_flow
from tests.e2e.test_mcp_client_simulation import run_mcp_protocol_session
from tests.integration.test_query_engine import _build_runtime
from tests.fakes import FakeLLM


def _ok_step(name: str, details: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "status": "passed", **details}


def _failed_step(name: str, exc: Exception) -> dict[str, Any]:
    return {"name": name, "status": "failed", "error": str(exc)}


def _run_ingestion_step(base_dir: Path) -> dict[str, Any]:
    base_dir.mkdir(parents=True, exist_ok=True)
    runtime = ServiceContainer(settings=AppSettings.model_validate({"environment": "test"}), services={})
    result = handle_ingest_documents(
        paths=[str(base_dir / "docs" / "a.md")],
        collection="acceptance-demo",
        runtime=runtime,
        pipeline_builder=lambda settings, collection=None: object(),
        source_discovery=lambda paths: ([Path(path) for path in paths], []),
        batch_runner=lambda pipeline, sources, collection, force_rebuild=False: [
            {
                "source_path": str(source),
                "result": {
                    "document_id": "doc-1",
                    "trace_id": "trace-ingest-acceptance",
                    "current_stage": "lifecycle_finalize",
                    "source_sha256": "sha-acceptance",
                    "smart_chunks": [{"chunk_id": "chunk-1"}],
                    "stored_ids": ["chunk-1"],
                    "status": "completed",
                    "lifecycle": {"final_status": "indexed"},
                },
            }
            for source in sources
        ],
    )
    payload = dict(result.structuredContent or {})
    return _ok_step(
        "ingestion",
        {
            "trace_id": payload.get("trace_id"),
            "artifact_paths": [str(base_dir / "docs" / "a.md")],
        },
    )


def _run_query_and_trace_step(base_dir: Path) -> dict[str, Any]:
    base_dir.mkdir(parents=True, exist_ok=True)
    settings, container = _build_runtime(base_dir / "query", reranker_provider=None, llm=FakeLLM(["RAG answer [1]."]))
    engine = build_query_engine(container, settings=settings)
    response = engine.run(
        query="what is rag",
        top_k=1,
        filters={"doc_type": "pdf"},
        trace_context={"trace_id": "trace-query-acceptance"},
    )
    trace = TraceService(settings).get_trace_detail("trace-query-acceptance")
    return _ok_step(
        "query_and_trace",
        {
            "trace_id": response["trace_id"],
            "answer": response["answer"],
            "trace_status": trace["status"],
            "artifact_paths": [str(settings.observability.log_file)],
        },
    )


def _run_mcp_step(base_dir: Path) -> dict[str, Any]:
    mcp_dir = base_dir / "mcp"
    mcp_dir.mkdir(parents=True, exist_ok=True)
    session = run_mcp_protocol_session(mcp_dir)
    return _ok_step(
        "mcp",
        {
            "trace_id": session["query"]["structuredContent"]["trace_id"],
            "run_id": session["evaluation"]["structuredContent"]["run_id"],
            "artifact_paths": [str(session["stderr_path"])],
        },
    )


def _run_dashboard_and_evaluation_step(base_dir: Path) -> dict[str, Any]:
    base_dir.mkdir(parents=True, exist_ok=True)
    dashboard_dir = base_dir / "dashboard"
    navigation_dir = base_dir / "navigation"
    evaluation_dir = base_dir / "evaluation"
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    navigation_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    smoke = dashboard_smoke_check(dashboard_dir)
    navigation = assert_dashboard_navigation_regression(navigation_dir)
    evaluation = assert_dashboard_regression_flow(evaluation_dir)
    page = evaluation["page"]
    return _ok_step(
        "dashboard_and_evaluation",
        {
            "pages": smoke,
            "navigation_targets": {
                "browser": navigation["browser"]["kind"],
                "query_compare": navigation["query_compare"]["kind"],
                "evaluation": navigation["evaluation"]["kind"],
            },
            "trace_id": page["selected_report"]["trace_id"],
            "run_id": page["selected_report"]["run_id"],
            "artifact_paths": [str((base_dir / "evaluation").resolve())],
        },
    )


def run_full_acceptance(base_dir: Path | None = None) -> dict[str, Any]:
    """Run the full local acceptance chain and return a structured summary."""

    temp_context = tempfile.TemporaryDirectory() if base_dir is None else None
    root = Path(temp_context.name) if temp_context is not None else Path(base_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "docs" / "a.md").write_text("alpha doc\n", encoding="utf-8")

    steps: list[dict[str, Any]] = []
    trace_ids: list[str] = []
    run_ids: list[str] = []
    artifact_paths: list[str] = []
    failed_steps: list[str] = []

    for name, runner in [
        ("ingestion", _run_ingestion_step),
        ("query_and_trace", _run_query_and_trace_step),
        ("mcp", _run_mcp_step),
        ("dashboard_and_evaluation", _run_dashboard_and_evaluation_step),
    ]:
        try:
            step = runner(root)
        except Exception as exc:  # pragma: no cover - unified acceptance boundary
            step = _failed_step(name, exc)
        steps.append(step)
        if step["status"] != "passed":
            failed_steps.append(name)
            continue
        if step.get("trace_id"):
            trace_ids.append(str(step["trace_id"]))
        if step.get("run_id"):
            run_ids.append(str(step["run_id"]))
        artifact_paths.extend(str(path) for path in step.get("artifact_paths") or [])

    summary = {
        "status": "passed" if not failed_steps else "failed",
        "failed_steps": failed_steps,
        "steps": steps,
        "artifact_paths": artifact_paths,
        "trace_ids": trace_ids,
        "run_ids": run_ids,
        "scenarios": {
            "scenario_1_data_preparation": {
                "status": "passed" if "ingestion" not in failed_steps else "failed",
                "steps": ["ingestion", "dashboard_and_evaluation"],
            },
            "scenario_2_recall_quality_evaluation": {
                "status": "passed" if not {"query_and_trace", "dashboard_and_evaluation"} & set(failed_steps) else "failed",
                "steps": ["query_and_trace", "dashboard_and_evaluation"],
            },
            "scenario_3_mcp_client_function": {
                "status": "passed" if "mcp" not in failed_steps else "failed",
                "steps": ["mcp"],
            },
        },
        "debug_hints": [
            "检查 artifact_paths 指向的日志、stderr 和临时数据目录。",
            "优先查看 failed_steps 对应的 trace_id 或 run_id。",
        ],
    }
    if temp_context is not None:
        summary["artifact_paths"].append(root.as_posix())
    return summary


def render_acceptance_summary(summary: dict[str, Any]) -> str:
    """Render the acceptance summary as stable JSON."""

    return json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the full RagMS acceptance chain.")
    parser.add_argument("--output", default=None, help="Optional file path for the JSON summary.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    summary = run_full_acceptance()
    rendered = render_acceptance_summary(summary)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
