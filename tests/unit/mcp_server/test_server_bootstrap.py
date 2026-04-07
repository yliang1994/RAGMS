from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from ragms.mcp_server.server import create_server, handle_initialize
from ragms.runtime.container import bootstrap_mcp_runtime


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
              collection: e1-tests
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


def test_create_server_exposes_initialize_metadata(tmp_path: Path) -> None:
    settings_path = write_settings(tmp_path / "settings.yaml")
    runtime = bootstrap_mcp_runtime(settings_path)

    server = create_server(runtime, log_level="INFO")
    init_options = handle_initialize(server)

    assert init_options.server_name == "ragms"
    assert init_options.capabilities.tools is not None
    assert init_options.capabilities.tools.listChanged is False
    assert getattr(server, "runtime_container") is runtime


@pytest.mark.anyio
async def test_stdio_server_initializes_without_stdout_pollution(tmp_path: Path) -> None:
    settings_path = write_settings(tmp_path / "settings.yaml")
    stderr_path = tmp_path / "server-stderr.log"
    repo_root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    pythonpath = str(repo_root / "src")
    env["PYTHONPATH"] = pythonpath if not env.get("PYTHONPATH") else f"{pythonpath}:{env['PYTHONPATH']}"

    server_params = StdioServerParameters(
        command=str(repo_root / ".venv" / "bin" / "python"),
        args=["-u", "scripts/run_mcp_server.py", "--settings", str(settings_path)],
        cwd=str(repo_root),
        env=env,
    )

    with stderr_path.open("w+", encoding="utf-8") as stderr_file:
        async with stdio_client(server_params, errlog=stderr_file) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                result = await session.initialize()

    assert result.serverInfo.name == "ragms"
    assert result.capabilities.tools is not None
    assert "Starting RagMS MCP server" in stderr_path.read_text(encoding="utf-8")


@pytest.mark.anyio
async def test_stdio_server_lists_tools_and_handles_a_placeholder_tool(tmp_path: Path) -> None:
    settings_path = write_settings(tmp_path / "settings.yaml")
    stderr_path = tmp_path / "server-stderr.log"
    repo_root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    pythonpath = str(repo_root / "src")
    env["PYTHONPATH"] = pythonpath if not env.get("PYTHONPATH") else f"{pythonpath}:{env['PYTHONPATH']}"

    server_params = StdioServerParameters(
        command=str(repo_root / ".venv" / "bin" / "python"),
        args=["-u", "scripts/run_mcp_server.py", "--settings", str(settings_path)],
        cwd=str(repo_root),
        env=env,
    )

    with stderr_path.open("w+", encoding="utf-8") as stderr_file:
        async with stdio_client(server_params, errlog=stderr_file) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools = await session.list_tools()
                result = await session.call_tool("get_trace_detail", {"trace_id": "trace-1"})

    assert [tool.name for tool in tools.tools] == [
        "query_knowledge_hub",
        "list_collections",
        "get_document_summary",
        "ingest_documents",
        "get_trace_detail",
        "evaluate_collection",
    ]
    assert result.isError is False
    assert result.content[0].type == "text"
    assert result.content[0].text == "get_trace_detail is not implemented yet"
