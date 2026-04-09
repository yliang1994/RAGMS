from __future__ import annotations

import json
import os
import select
import subprocess
import textwrap
import time
from pathlib import Path
from typing import Any

from mcp import types


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _base_env() -> dict[str, str]:
    repo_root = _repo_root()
    env = os.environ.copy()
    pythonpath = str(repo_root / "src")
    env["PYTHONPATH"] = pythonpath if not env.get("PYTHONPATH") else f"{pythonpath}:{env['PYTHONPATH']}"
    return env


def _write_protocol_server(script_path: Path) -> Path:
    script_path.write_text(
        textwrap.dedent(
            """
            from __future__ import annotations

            import json
            import sys
            from pathlib import Path

            from mcp import types

            from ragms.core.evaluation import ReportService
            from ragms.mcp_server.protocol_handler import JSONRPC_INTERNAL_ERROR, JSONRPC_METHOD_NOT_FOUND
            from ragms.mcp_server.schemas import get_input_schema
            from ragms.mcp_server.tools.collections import handle_list_collections
            from ragms.mcp_server.tools.documents import handle_get_document_summary
            from ragms.mcp_server.tools.evaluation import handle_evaluate_collection
            from ragms.mcp_server.tools.ingest import handle_ingest_documents
            from ragms.mcp_server.tools.query import handle_query_knowledge_hub
            from ragms.mcp_server.tools.traces import handle_get_trace_detail
            from ragms.runtime.container import ServiceContainer
            from ragms.runtime.settings_models import AppSettings
            from ragms.storage.sqlite.schema import initialize_metadata_schema


            class StubDataService:
                def list_collections(self, filters=None, page=None, page_size=None):
                    return {
                        "collections": [
                            {
                                "name": "alpha",
                                "document_count": 1,
                                "chunk_count": 2,
                                "image_count": 0,
                                "latest_updated_at": "2026-04-09T00:00:00Z",
                            }
                        ],
                        "pagination": {
                            "page": 1,
                            "page_size": page_size,
                            "total_count": 1,
                            "returned_count": 1,
                            "has_more": False,
                        },
                    }

                def get_document_summary(self, document_id):
                    if document_id != "doc-1":
                        from ragms.core.management.data_service import DocumentSummaryNotFoundError

                        raise DocumentSummaryNotFoundError(f"Document not found: {document_id}")
                    return {
                        "document_id": "doc-1",
                        "source_path": "docs/a.md",
                        "primary_collection": "alpha",
                        "collections": ["alpha"],
                        "summary": "alpha summary",
                        "structure_outline": ["Intro"],
                        "key_metadata": {"title": "Alpha"},
                        "ingestion_status": {"status": "indexed"},
                        "page_summary": {"pages": [1], "page_count": 1},
                        "image_summary": {"image_count": 0, "images": []},
                        "chunk_count": 2,
                    }


            class StubTraceService:
                def get_trace_detail(self, trace_id):
                    if trace_id != "trace-query-1":
                        from ragms.core.management.trace_service import TraceNotFoundError

                        raise TraceNotFoundError(f"Trace not found: {trace_id}")
                    return {
                        "trace_id": "trace-query-1",
                        "trace_type": "query",
                        "status": "succeeded",
                        "started_at": "2026-04-09T00:00:00Z",
                        "finished_at": "2026-04-09T00:00:01Z",
                        "duration_ms": 100,
                        "collection": "alpha",
                        "metadata": {},
                        "error": None,
                        "stages": [{"stage_name": "query_processing", "status": "succeeded", "metadata": {}, "error": None}],
                    }


            class StubQueryEngine:
                def run(self, **kwargs):
                    return {
                        "markdown": "RAG answer [1]",
                        "answer": "RAG answer [1]",
                        "content": [types.TextContent(type="text", text="RAG answer [1]")],
                        "structured_content": {
                            "trace_id": "trace-query-1",
                            "citations": [{"index": 1, "chunk_id": "chunk-1", "source_path": "docs/a.md"}],
                            "retrieved_chunks": [{"chunk_id": "chunk-1", "content": "alpha chunk"}],
                        },
                    }


            class StubEvalRunner:
                def run(self, **kwargs):
                    return {
                        "run_id": "run-1",
                        "trace_id": "trace-eval-1",
                        "collection": kwargs.get("collection") or "alpha",
                        "dataset_name": kwargs.get("dataset_name") or "golden",
                        "dataset_version": kwargs.get("dataset_version") or "v1",
                        "backend_set": list(kwargs.get("backend_set") or ["custom_metrics"]),
                        "aggregate_metrics": {"hit_rate_at_k": 0.95, "mrr": 0.84, "ndcg_at_k": 0.9},
                        "quality_gate_status": "passed",
                        "failed_samples": [{"sample_id": "sample-2", "stage": "sample_build", "error": {"message": "missing"}}],
                        "path": str(runtime.settings.paths.data_dir / "evaluation" / "reports" / "run-1.json"),
                    }


            settings = AppSettings.model_validate({"environment": "test"})
            settings = settings.model_copy(deep=True)
            settings.app_name = "ragms-protocol-test"
            settings.paths.project_root = Path(".").resolve()
            settings.paths.data_dir = (Path.cwd() / "data").resolve()
            settings.paths.logs_dir = (Path.cwd() / "logs").resolve()
            settings.storage.sqlite.path = settings.paths.data_dir / "metadata" / "ragms.db"
            settings.observability.log_file = settings.paths.logs_dir / "traces.jsonl"
            settings.dashboard.traces_file = settings.paths.logs_dir / "traces.jsonl"
            settings.vector_store.collection = "alpha"
            settings.paths.data_dir.mkdir(parents=True, exist_ok=True)
            settings.paths.logs_dir.mkdir(parents=True, exist_ok=True)
            connection = initialize_metadata_schema(settings.storage.sqlite.path)
            report_service = ReportService(settings, connection=connection)
            report_service.write_report(
                {
                    "run_id": "baseline-run",
                    "trace_id": "trace-baseline",
                    "collection": "alpha",
                    "dataset_name": "golden",
                    "dataset_version": "v1",
                    "backend_set": ["custom_metrics"],
                    "aggregate_metrics": {"hit_rate_at_k": 0.8, "mrr": 0.7, "ndcg_at_k": 0.85},
                    "quality_gate_status": "failed",
                    "config_snapshot": {},
                    "samples": [{"sample_id": "sample-1", "metrics_summary": {"hit_rate_at_k": 0.8}}],
                    "failed_samples": [],
                }
            )
            report_service.set_baseline("baseline-run")
            report_service.write_report(
                {
                    "run_id": "run-1",
                    "trace_id": "trace-eval-1",
                    "collection": "alpha",
                    "dataset_name": "golden",
                    "dataset_version": "v1",
                    "backend_set": ["custom_metrics"],
                    "aggregate_metrics": {"hit_rate_at_k": 0.95, "mrr": 0.84, "ndcg_at_k": 0.9},
                    "quality_gate_status": "passed",
                    "config_snapshot": {},
                    "samples": [{"sample_id": "sample-1", "metrics_summary": {"hit_rate_at_k": 0.95}}],
                    "failed_samples": [{"sample_id": "sample-2", "stage": "sample_build", "error": {"message": "missing"}}],
                }
            )
            runtime = ServiceContainer(settings=settings, services={})
            data_service = StubDataService()
            trace_service = StubTraceService()
            eval_runner = StubEvalRunner()


            registry = {
                "query_knowledge_hub": {
                    "description": "query",
                    "inputSchema": get_input_schema("query_knowledge_hub"),
                    "handler": lambda arguments: handle_query_knowledge_hub(
                        query=arguments.get("query"),
                        collection=arguments.get("collection"),
                        top_k=arguments.get("top_k", 5),
                        filters=arguments.get("filters"),
                        return_debug=arguments.get("return_debug", False),
                        runtime=runtime,
                        query_engine=StubQueryEngine(),
                    ),
                },
                "list_collections": {
                    "description": "collections",
                    "inputSchema": get_input_schema("list_collections"),
                    "handler": lambda arguments: handle_list_collections(
                        filters=arguments.get("filters"),
                        page=arguments.get("page"),
                        page_size=arguments.get("page_size"),
                        runtime=runtime,
                        data_service=data_service,
                    ),
                },
                "get_document_summary": {
                    "description": "documents",
                    "inputSchema": get_input_schema("get_document_summary"),
                    "handler": lambda arguments: handle_get_document_summary(
                        document_id=arguments.get("document_id"),
                        runtime=runtime,
                        data_service=data_service,
                    ),
                },
                "ingest_documents": {
                    "description": "ingest",
                    "inputSchema": get_input_schema("ingest_documents"),
                    "handler": lambda arguments: handle_ingest_documents(
                        paths=arguments.get("paths"),
                        collection=arguments.get("collection"),
                        force_rebuild=arguments.get("force_rebuild", False),
                        options=arguments.get("options"),
                        runtime=runtime,
                        pipeline_builder=lambda settings, collection=None: object(),
                        source_discovery=lambda paths: ([Path(path) for path in paths], []),
                        batch_runner=lambda pipeline, sources, collection, force_rebuild=False: [
                            {
                                "source_path": str(source),
                                "result": {
                                    "document_id": "doc-1",
                                    "trace_id": "trace-ingest-1",
                                    "current_stage": "lifecycle_finalize",
                                    "source_sha256": "sha-1",
                                    "smart_chunks": [{"chunk_id": "chunk-1"}],
                                    "stored_ids": ["chunk-1"],
                                    "status": "completed",
                                    "lifecycle": {"final_status": "indexed"},
                                },
                            }
                            for source in sources
                        ],
                    ),
                },
                "get_trace_detail": {
                    "description": "trace",
                    "inputSchema": get_input_schema("get_trace_detail"),
                    "handler": lambda arguments: handle_get_trace_detail(
                        trace_id=arguments.get("trace_id"),
                        runtime=runtime,
                        trace_service=trace_service,
                    ),
                },
                "evaluate_collection": {
                    "description": "evaluate",
                    "inputSchema": get_input_schema("evaluate_collection"),
                    "handler": lambda arguments: handle_evaluate_collection(
                        collection=arguments.get("collection"),
                        dataset=arguments.get("dataset"),
                        metrics=arguments.get("metrics"),
                        eval_options=arguments.get("eval_options"),
                        baseline_mode=arguments.get("baseline_mode", "compare"),
                        runtime=runtime,
                        eval_runner=eval_runner,
                        report_service=report_service,
                    ),
                },
            }

            sys.stderr.write("INFO:ragms.mcp_server.server:Starting RagMS MCP server name=ragms-protocol-test version=1.0.0 tools_capability=True\\n")
            sys.stderr.flush()
            initialized = False
            for raw_line in sys.stdin:
                message = json.loads(raw_line)
                method = message.get("method")
                msg_id = message.get("id")
                params = dict(message.get("params") or {})
                if method == "initialize":
                    response = {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {
                            "protocolVersion": types.LATEST_PROTOCOL_VERSION,
                            "capabilities": {"tools": {"listChanged": False}},
                            "serverInfo": {"name": "ragms-protocol-test", "version": "1.0.0"},
                            "instructions": "protocol simulation server",
                        },
                    }
                elif method == "notifications/initialized":
                    initialized = True
                    continue
                elif method == "tools/list":
                    response = {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {
                            "tools": [
                                {
                                    "name": name,
                                    "description": payload["description"],
                                    "inputSchema": payload["inputSchema"],
                                }
                                for name, payload in registry.items()
                            ]
                        },
                    }
                elif method == "tools/call":
                    if not initialized:
                        response = {
                            "jsonrpc": "2.0",
                            "id": msg_id,
                            "error": {"code": -32602, "message": "Received request before initialization was complete"},
                        }
                    else:
                        tool_name = params.get("name")
                        if tool_name not in registry:
                            response = {
                                "jsonrpc": "2.0",
                                "id": msg_id,
                                "error": {"code": JSONRPC_METHOD_NOT_FOUND, "message": f"Unknown tool: {tool_name}"},
                            }
                        else:
                            try:
                                result = registry[tool_name]["handler"](dict(params.get("arguments") or {}))
                                if result.isError and result.structuredContent and "error" in result.structuredContent:
                                    response = {"jsonrpc": "2.0", "id": msg_id, "result": result.model_dump(by_alias=True, exclude_none=True)}
                                else:
                                    response = {"jsonrpc": "2.0", "id": msg_id, "result": result.model_dump(by_alias=True, exclude_none=True)}
                            except Exception as exc:
                                response = {
                                    "jsonrpc": "2.0",
                                    "id": msg_id,
                                    "error": {"code": JSONRPC_INTERNAL_ERROR, "message": str(exc)},
                                }
                else:
                    response = {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {"code": JSONRPC_METHOD_NOT_FOUND, "message": f"Unknown method: {method}"},
                    }
                sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\\n")
                sys.stdout.flush()
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return script_path


def _start_protocol_server(tmp_path: Path) -> tuple[subprocess.Popen[bytes], Path]:
    repo_root = _repo_root()
    stderr_path = tmp_path / "mcp-client-simulation.stderr.log"
    server_script = _write_protocol_server(tmp_path / "protocol_server.py")
    stderr_file = stderr_path.open("wb")
    process = subprocess.Popen(
        [str(repo_root / ".venv" / "bin" / "python"), "-u", str(server_script)],
        cwd=tmp_path,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr_file,
        env=_base_env(),
    )
    return process, stderr_path


def _send_message(process: subprocess.Popen[bytes], payload: dict[str, Any]) -> None:
    if process.stdin is None:
        raise AssertionError("stdin unavailable")
    process.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
    process.stdin.flush()


def _read_message(process: subprocess.Popen[bytes], *, timeout_seconds: float = 5.0) -> dict[str, Any]:
    if process.stdout is None:
        raise AssertionError("stdout unavailable")
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        ready, _, _ = select.select([process.stdout], [], [], 0.2)
        if process.stdout not in ready:
            continue
        raw_line = process.stdout.readline()
        if not raw_line:
            break
        return json.loads(raw_line.decode("utf-8"))
    raise AssertionError("Timed out waiting for MCP response")


def assert_mcp_tool_roundtrip(
    process: subprocess.Popen[bytes],
    *,
    request_id: int,
    name: str,
    arguments: dict[str, Any] | None = None,
    expect_error: bool = False,
) -> dict[str, Any]:
    """Call one tool over JSON-RPC and assert the basic contract."""

    _send_message(
        process,
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments or {},
            },
        },
    )
    response = _read_message(process)
    if "error" in response:
        assert expect_error is True
        assert response["error"]["code"] < 0
        return response
    result = dict(response["result"])
    assert bool(result["isError"]) is expect_error
    assert result["content"][0]["type"] == "text"
    if expect_error:
        assert result["structuredContent"]["error"]["code"] < 0
    else:
        assert isinstance(result.get("structuredContent"), dict)
    return result


def run_mcp_protocol_session(tmp_path: Path) -> dict[str, Any]:
    """Run a full MCP protocol session against a real child-process server."""

    process, stderr_path = _start_protocol_server(tmp_path)
    try:
        _send_message(
            process,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": types.LATEST_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "pytest", "version": "1.0"},
                },
            },
        )
        initialize = _read_message(process)
        _send_message(process, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        _send_message(process, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        tools = _read_message(process)
        query = assert_mcp_tool_roundtrip(
            process,
            request_id=3,
            name="query_knowledge_hub",
            arguments={"query": "what is rag", "top_k": 1},
        )
        collections = assert_mcp_tool_roundtrip(process, request_id=4, name="list_collections", arguments={})
        document = assert_mcp_tool_roundtrip(
            process,
            request_id=5,
            name="get_document_summary",
            arguments={"document_id": "doc-1"},
        )
        ingest = assert_mcp_tool_roundtrip(
            process,
            request_id=6,
            name="ingest_documents",
            arguments={"paths": ["docs/a.md"]},
        )
        trace = assert_mcp_tool_roundtrip(
            process,
            request_id=7,
            name="get_trace_detail",
            arguments={"trace_id": "trace-query-1"},
        )
        evaluation = assert_mcp_tool_roundtrip(
            process,
            request_id=8,
            name="evaluate_collection",
            arguments={"collection": "alpha", "dataset": "golden", "eval_options": {"dataset_version": "v1", "backend_set": ["custom_metrics"]}},
        )
        invalid = assert_mcp_tool_roundtrip(
            process,
            request_id=9,
            name="query_knowledge_hub",
            arguments={"query": "what is rag", "top_k": 0},
            expect_error=True,
        )
    finally:
        process.kill()
        process.wait()
    return {
        "initialize": initialize,
        "tools": tools,
        "query": query,
        "collections": collections,
        "document": document,
        "ingest": ingest,
        "trace": trace,
        "evaluation": evaluation,
        "invalid": invalid,
        "stderr_path": stderr_path,
    }


def test_mcp_client_simulation_covers_handshake_tool_roundtrips_and_error_semantics(tmp_path: Path) -> None:
    session = run_mcp_protocol_session(tmp_path)

    assert session["initialize"]["result"]["serverInfo"]["name"] == "ragms-protocol-test"
    assert [tool["name"] for tool in session["tools"]["result"]["tools"]] == [
        "query_knowledge_hub",
        "list_collections",
        "get_document_summary",
        "ingest_documents",
        "get_trace_detail",
        "evaluate_collection",
    ]
    assert session["query"]["structuredContent"]["trace_id"] == "trace-query-1"
    assert session["collections"]["structuredContent"]["collections"][0]["name"] == "alpha"
    assert session["document"]["structuredContent"]["document_id"] == "doc-1"
    assert session["ingest"]["structuredContent"]["trace_id"] == "trace-ingest-1"
    assert session["trace"]["structuredContent"]["trace_id"] == "trace-query-1"
    assert session["evaluation"]["structuredContent"]["run_id"] == "run-1"
    assert session["evaluation"]["structuredContent"]["baseline_delta"]["hit_rate_at_k"] == 0.15
    assert session["evaluation"]["structuredContent"]["failed_samples_count"] == 1
    assert session["invalid"]["error"]["message"] == "Invalid arguments for query_knowledge_hub.top_k: Input should be greater than or equal to 1"
    assert "Starting RagMS MCP server" in session["stderr_path"].read_text(encoding="utf-8")
