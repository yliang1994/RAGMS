from __future__ import annotations

import base64
import json
import os
import select
import subprocess
import textwrap
import time
from pathlib import Path
from typing import Any

from mcp import types


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
              api_key: null
              base_url: null
            embedding:
              provider: openai
              model: text-embedding-3-small
              api_key: null
              base_url: null
              batch_size: 8
            ingestion:
              transform:
                enable_llm_chunk_refine: false
                enable_llm_metadata_enrich: false
            vision_llm:
              provider: auto
              model: gpt-4.1-mini
              api_key: null
              base_url: null
              language_providers:
                zh: qwen_vl
                en: gpt4o
              environment_providers:
                development: qwen_vl
                test: qwen_vl
                production: gpt4o
            vector_store:
              backend: chroma
              collection: e9-tests
            storage:
              sqlite:
                path: data/metadata/ragms.db
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
              enabled: false
              port: 8501
              traces_file: logs/traces.jsonl
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _base_env() -> dict[str, str]:
    repo_root = _repo_root()
    env = os.environ.copy()
    pythonpath = str(repo_root / "src")
    env["PYTHONPATH"] = pythonpath if not env.get("PYTHONPATH") else f"{pythonpath}:{env['PYTHONPATH']}"
    return env


def _write_manual_stdio_server(script_path: Path) -> Path:
    image_bytes = base64.b64encode(b"fake-image-bytes").decode("ascii")
    script_path.write_text(
        textwrap.dedent(
            f"""
            from __future__ import annotations

            import json
            import sys

            latest_protocol = {types.LATEST_PROTOCOL_VERSION!r}
            tools = [
                {{
                    "name": "query_knowledge_hub",
                    "description": "query",
                    "inputSchema": {{"type": "object", "properties": {{"query": {{"type": "string"}}}}, "required": ["query"]}},
                }},
                {{
                    "name": "list_collections",
                    "description": "collections",
                    "inputSchema": {{"type": "object", "properties": {{}}}},
                }},
                {{
                    "name": "get_document_summary",
                    "description": "documents",
                    "inputSchema": {{"type": "object", "properties": {{"document_id": {{"type": "string"}}}}, "required": ["document_id"]}},
                }},
                {{
                    "name": "ingest_documents",
                    "description": "ingest",
                    "inputSchema": {{"type": "object", "properties": {{"paths": {{"type": "array"}}}}, "required": ["paths"]}},
                }},
            ]


            def respond(message: dict) -> dict:
                method = message.get("method")
                msg_id = message.get("id")
                params = dict(message.get("params") or {{}})
                if method == "initialize":
                    return {{
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {{
                            "protocolVersion": latest_protocol,
                            "capabilities": {{"tools": {{"listChanged": False}}}},
                            "serverInfo": {{"name": "ragms-stdio-test", "version": "1.0.0"}},
                            "instructions": "manual stdio test server",
                        }},
                    }}
                if method == "notifications/initialized":
                    return {{}}
                if method == "tools/list":
                    return {{
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {{"tools": tools}},
                    }}
                if method == "tools/call":
                    tool_name = params.get("name")
                    arguments = dict(params.get("arguments") or {{}})
                    if tool_name == "query_knowledge_hub":
                        query = arguments.get("query")
                        if query == "empty":
                            result = {{
                                "content": [{{"type": "text", "text": "No relevant results found."}}],
                                "structuredContent": {{"trace_id": "trace-empty", "citations": [], "retrieved_chunks": []}},
                                "isError": False,
                            }}
                        elif query == "boom":
                            result = {{
                                "content": [{{"type": "text", "text": "simulated query failure"}}],
                                "structuredContent": {{"error": {{"code": -32603, "message": "simulated query failure"}}}},
                                "isError": True,
                            }}
                        else:
                            result = {{
                                "content": [
                                    {{"type": "text", "text": "RAG answer [1]"}},
                                    {{"type": "image", "mimeType": "image/png", "data": "{image_bytes}"}},
                                ],
                                "structuredContent": {{
                                    "trace_id": "trace-query",
                                    "citations": [{{"index": 1, "chunk_id": "chunk-1", "source_path": "docs/rag.pdf"}}],
                                    "retrieved_chunks": [{{"chunk_id": "chunk-1"}}],
                                }},
                                "isError": False,
                            }}
                    elif tool_name == "list_collections":
                        result = {{
                            "content": [{{"type": "text", "text": "Found 1 collection(s)."}}],
                            "structuredContent": {{
                                "collections": [
                                    {{
                                        "name": "alpha",
                                        "document_count": 2,
                                        "chunk_count": 4,
                                        "image_count": 1,
                                        "latest_updated_at": "2026-04-07T00:00:00+00:00",
                                    }}
                                ],
                                "pagination": {{"page": 1, "page_size": None, "total_count": 1, "returned_count": 1, "has_more": False}},
                                "summary": {{"collection_count": 1, "filters_applied": {{}}}},
                            }},
                            "isError": False,
                        }}
                    elif tool_name == "get_document_summary":
                        result = {{
                            "content": [{{"type": "text", "text": "Document summary"}}],
                            "structuredContent": {{
                                "document_id": arguments.get("document_id"),
                                "source_path": "docs/a.md",
                                "primary_collection": "alpha",
                                "collections": ["alpha"],
                                "summary": "Document summary",
                                "structure_outline": ["Intro"],
                                "key_metadata": {{"title": "Alpha"}},
                                "ingestion_status": {{"status": "indexed"}},
                                "page_summary": {{"pages": [1], "page_count": 1}},
                                "image_summary": {{"image_count": 0, "images": []}},
                                "chunk_count": 1,
                            }},
                            "isError": False,
                        }}
                    elif tool_name == "ingest_documents":
                        result = {{
                            "content": [{{"type": "text", "text": "Ingestion accepted 1 source(s): indexed 1, skipped 0, failed 0."}}],
                            "structuredContent": {{
                                "trace_id": "trace-ingest",
                                "collection": "alpha",
                                "requested_paths": list(arguments.get("paths") or []),
                                "summary": {{
                                    "requested_path_count": 1,
                                    "resolved_source_count": 1,
                                    "accepted_count": 1,
                                    "document_count": 1,
                                    "indexed_count": 1,
                                    "skipped_count": 0,
                                    "failed_count": 0,
                                }},
                                "documents": [
                                    {{
                                        "source_path": "docs/a.md",
                                        "collection": "alpha",
                                        "document_id": "doc-1",
                                        "trace_id": "trace-ingest",
                                        "status": "indexed",
                                        "current_stage": "lifecycle",
                                        "source_sha256": "sha-1",
                                        "chunk_count": 2,
                                        "stored_count": 2,
                                        "skipped": False,
                                        "error": None,
                                    }}
                                ],
                                "skipped_summary": {{"count": 0, "documents": []}},
                                "failure_summary": {{"count": 0, "documents": []}},
                            }},
                            "isError": False,
                        }}
                    else:
                        result = {{
                            "content": [{{"type": "text", "text": "Unknown tool"}}],
                            "structuredContent": {{"error": {{"code": -32601, "message": "Unknown tool"}}}},
                            "isError": True,
                        }}
                    return {{"jsonrpc": "2.0", "id": msg_id, "result": result}}
                return {{
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {{"code": -32601, "message": f"Unknown method: {{method}}"}},
                }}


            for raw_line in sys.stdin:
                payload = json.loads(raw_line)
                response = respond(payload)
                if response:
                    sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\\n")
                    sys.stdout.flush()
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return script_path


def start_stdio_server_process(tmp_path: Path, *, scripted: bool) -> tuple[subprocess.Popen[bytes], Path]:
    """Start a real stdio subprocess and return the process handle plus stderr path."""

    repo_root = _repo_root()
    stderr_path = tmp_path / ("stdio-scripted.stderr.log" if scripted else "stdio-real.stderr.log")
    if scripted:
        server_script = _write_manual_stdio_server(tmp_path / "manual_stdio_server.py")
        command = [str(repo_root / ".venv" / "bin" / "python"), "-u", str(server_script)]
    else:
        settings_path = _write_settings(tmp_path / "settings.yaml")
        command = [
            str(repo_root / ".venv" / "bin" / "python"),
            "-u",
            "scripts/run_mcp_server.py",
            "--settings",
            str(settings_path),
        ]

    stderr_file = stderr_path.open("wb")
    process = subprocess.Popen(
        command,
        cwd=repo_root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr_file,
        env=_base_env(),
    )
    return process, stderr_path


def _send_message(process: subprocess.Popen[bytes], payload: dict[str, Any]) -> None:
    if process.stdin is None:
        raise AssertionError("stdio process stdin is not available")
    process.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
    process.stdin.flush()


def _read_message(process: subprocess.Popen[bytes], *, timeout_seconds: float = 5.0) -> dict[str, Any]:
    if process.stdout is None:
        raise AssertionError("stdio process stdout is not available")

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        ready, _, _ = select.select([process.stdout], [], [], 0.2)
        if process.stdout not in ready:
            continue
        raw_line = process.stdout.readline()
        if not raw_line:
            break
        return json.loads(raw_line.decode("utf-8"))
    raise AssertionError("Timed out waiting for stdio MCP response")


def call_mcp_tool(
    process: subprocess.Popen[bytes],
    *,
    request_id: int,
    name: str,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call one MCP tool over raw stdio JSON-RPC and return the result payload."""

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
    return dict(response["result"])


def assert_mcp_response_contract(
    result: dict[str, Any],
    *,
    expect_error: bool = False,
    require_image: bool = False,
) -> None:
    """Assert the shared MCP tool response contract on raw JSON-RPC payloads."""

    assert bool(result["isError"]) is expect_error
    content = list(result.get("content") or [])
    assert content
    assert content[0]["type"] == "text"
    if expect_error:
        assert result["structuredContent"]["error"]["code"] < 0
        return

    assert isinstance(result.get("structuredContent"), dict)
    if require_image:
        assert any(item.get("type") == "image" for item in content)


def test_stdio_jsonrpc_flow_supports_initialize_list_and_core_tool_calls(tmp_path: Path) -> None:
    process, stderr_path = start_stdio_server_process(tmp_path, scripted=True)
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
        query = call_mcp_tool(process, request_id=3, name="query_knowledge_hub", arguments={"query": "with-image"})
        empty = call_mcp_tool(process, request_id=4, name="query_knowledge_hub", arguments={"query": "empty"})
        collections = call_mcp_tool(process, request_id=5, name="list_collections", arguments={})
        document = call_mcp_tool(process, request_id=6, name="get_document_summary", arguments={"document_id": "doc-1"})
        ingest = call_mcp_tool(process, request_id=7, name="ingest_documents", arguments={"paths": ["docs/a.md"]})
    finally:
        process.kill()
        process.wait()

    assert initialize["result"]["serverInfo"]["name"] == "ragms-stdio-test"
    assert [tool["name"] for tool in tools["result"]["tools"]] == [
        "query_knowledge_hub",
        "list_collections",
        "get_document_summary",
        "ingest_documents",
    ]
    assert_mcp_response_contract(query, require_image=True)
    assert query["structuredContent"]["citations"][0]["chunk_id"] == "chunk-1"
    assert_mcp_response_contract(empty)
    assert empty["structuredContent"]["citations"] == []
    assert_mcp_response_contract(collections)
    assert collections["structuredContent"]["collections"][0]["name"] == "alpha"
    assert_mcp_response_contract(document)
    assert document["structuredContent"]["document_id"] == "doc-1"
    assert_mcp_response_contract(ingest)
    assert ingest["structuredContent"]["summary"]["indexed_count"] == 1
    assert stderr_path.read_text(encoding="utf-8") == ""


def test_real_stdio_server_routes_logs_to_stderr_without_stdout_pollution(tmp_path: Path) -> None:
    process, stderr_path = start_stdio_server_process(tmp_path, scripted=False)
    try:
        if process.stdout is None:
            raise AssertionError("real stdio process stdout is not available")
        ready, _, _ = select.select([process.stdout], [], [], 1.0)
        assert process.stdout not in ready
        stderr_text = ""
        deadline = time.time() + 5.0
        while time.time() < deadline:
            stderr_text = stderr_path.read_text(encoding="utf-8")
            if "Starting RagMS MCP server" in stderr_text:
                break
            time.sleep(0.1)
    finally:
        process.kill()
        process.wait()

    assert "Starting RagMS MCP server" in stderr_text
