"""MCP server bootstrap and STDIO runtime entrypoints."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mcp import types
from mcp.server.fastmcp import FastMCP
from mcp.server.models import InitializationOptions
from mcp.shared.version import SUPPORTED_PROTOCOL_VERSIONS

from ragms.mcp_server.protocol_handler import JSONRPC_INTERNAL_ERROR, JSONRPC_INVALID_PARAMS, JSONRPC_METHOD_NOT_FOUND, ProtocolHandler
from ragms.mcp_server.tool_registry import build_tool_registry, list_tool_definitions, register_tools
from ragms.runtime.container import ServiceContainer, bootstrap_mcp_runtime


LOGGER = logging.getLogger(__name__)


def _configure_stderr_logging(log_level: str) -> None:
    """Route application bootstrap logs to stderr only."""

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(levelname)s:%(name)s:%(message)s",
        stream=sys.stderr,
        force=True,
    )


def create_server(
    runtime: ServiceContainer,
    *,
    log_level: str = "INFO",
) -> FastMCP:
    """Create the MCP server instance for the provided runtime container."""

    server = FastMCP(
        name=runtime.settings.app_name,
        instructions=(
            "RagMS local MCP server. "
            "This bootstrap currently exposes initialization and empty tool discovery only."
        ),
        log_level=log_level.upper(),
    )
    tool_registry = build_tool_registry(runtime=runtime)
    register_tools(server, tool_registry)
    setattr(server, "runtime_container", runtime)
    setattr(server, "tool_registry", tool_registry)
    return server


def handle_initialize(server: FastMCP) -> InitializationOptions:
    """Return the initialization payload that the server will expose to clients."""

    return server._mcp_server.create_initialization_options()


def _initialization_payload(
    init_options: InitializationOptions,
    *,
    requested_version: str | int | None,
) -> dict[str, Any]:
    if requested_version in SUPPORTED_PROTOCOL_VERSIONS:
        protocol_version = requested_version
    else:
        protocol_version = types.LATEST_PROTOCOL_VERSION

    return {
        "protocolVersion": protocol_version,
        "capabilities": init_options.capabilities.model_dump(by_alias=True, exclude_none=True),
        "serverInfo": {
            "name": init_options.server_name,
            "version": init_options.server_version,
        },
        "instructions": init_options.instructions,
    }


def _tool_payloads(runtime: ServiceContainer) -> list[dict[str, Any]]:
    registry = build_tool_registry(runtime=runtime)
    return [
        {
            "name": definition.name,
            "title": definition.title,
            "description": definition.description,
            "inputSchema": definition.input_schema,
        }
        for definition in list_tool_definitions(registry)
    ]


def _write_jsonrpc_response(
    *,
    msg_id: Any,
    result: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {"jsonrpc": "2.0", "id": msg_id}
    if error is not None:
        payload["error"] = error
    else:
        payload["result"] = result or {}
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _write_jsonrpc_error(*, msg_id: Any, code: int, message: str, data: dict[str, Any] | None = None) -> None:
    error = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    _write_jsonrpc_response(msg_id=msg_id, error=error)


def _normalize_tool_result(result: Any) -> types.CallToolResult:
    if isinstance(result, types.CallToolResult):
        return result

    handler = ProtocolHandler()
    if isinstance(result, dict):
        return handler.build_success_response(
            text=json.dumps(result, ensure_ascii=False),
            structured_content=result,
        )
    return handler.build_success_response(text=str(result))


def _serve_stdio_jsonrpc(
    runtime: ServiceContainer,
    *,
    log_level: str = "INFO",
) -> None:
    """Run a minimal MCP stdio loop that is compatible with the official client."""

    _configure_stderr_logging(log_level)
    server = create_server(runtime, log_level=log_level)
    init_options = handle_initialize(server)
    registry = getattr(server, "tool_registry")
    initialized = False

    LOGGER.info(
        "Starting RagMS MCP server name=%s version=%s tools_capability=%s",
        init_options.server_name,
        init_options.server_version,
        bool(init_options.capabilities.tools),
    )

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            LOGGER.warning("Discarding invalid JSON-RPC line: %s", exc)
            continue

        msg_id = message.get("id")
        method = message.get("method")
        params = dict(message.get("params") or {})

        if method == "initialize":
            _write_jsonrpc_response(
                msg_id=msg_id,
                result=_initialization_payload(
                    init_options,
                    requested_version=params.get("protocolVersion"),
                ),
            )
            continue

        if method == "notifications/initialized":
            initialized = True
            continue

        if method == "tools/list":
            if not initialized:
                _write_jsonrpc_error(
                    msg_id=msg_id,
                    code=JSONRPC_INVALID_PARAMS,
                    message="Received request before initialization was complete",
                )
                continue
            _write_jsonrpc_response(msg_id=msg_id, result={"tools": _tool_payloads(runtime)})
            continue

        if method == "tools/call":
            if not initialized:
                _write_jsonrpc_error(
                    msg_id=msg_id,
                    code=JSONRPC_INVALID_PARAMS,
                    message="Received request before initialization was complete",
                )
                continue

            tool_name = params.get("name")
            arguments = dict(params.get("arguments") or {})
            definition = registry.get(tool_name)
            if definition is None:
                _write_jsonrpc_error(
                    msg_id=msg_id,
                    code=JSONRPC_METHOD_NOT_FOUND,
                    message=f"Unknown tool: {tool_name}",
                )
                continue

            try:
                raw_result = definition.handler(**arguments)
                normalized_result = _normalize_tool_result(raw_result)
            except TypeError as exc:
                _write_jsonrpc_error(
                    msg_id=msg_id,
                    code=JSONRPC_INVALID_PARAMS,
                    message=str(exc) or "Invalid parameters",
                )
                continue
            except Exception as exc:  # pragma: no cover - unified tool boundary
                LOGGER.exception("Tool execution failed: %s", tool_name)
                _write_jsonrpc_error(
                    msg_id=msg_id,
                    code=JSONRPC_INTERNAL_ERROR,
                    message=str(exc) or "Internal error",
                )
                continue

            _write_jsonrpc_response(
                msg_id=msg_id,
                result=normalized_result.model_dump(by_alias=True, exclude_none=True),
            )
            continue

        if method == "ping":
            _write_jsonrpc_response(msg_id=msg_id, result={})
            continue

        if msg_id is not None:
            _write_jsonrpc_error(
                msg_id=msg_id,
                code=JSONRPC_METHOD_NOT_FOUND,
                message=f"Unknown method: {method}",
            )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local RagMS MCP server.")
    parser.add_argument(
        "--settings",
        default="settings.yaml",
        help="Path to the settings.yaml file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level for MCP server bootstrap logs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate bootstrap and exit without starting the STDIO server.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _format_bootstrap_message(runtime: ServiceContainer) -> str:
    """Build a deterministic bootstrap summary for dry-run callers."""

    return (
        "MCP server bootstrap ready: "
        f"llm={runtime.get('llm').implementation} "
        f"embedding={runtime.get('embedding').implementation} "
        f"collection={runtime.settings.vector_store.collection}"
    )


def run_mcp_server_main(
    argv: Sequence[str] | None = None,
    *,
    serve: bool | None = None,
) -> int:
    """Bootstrap the MCP runtime and optionally enter STDIO serving mode."""

    args = _parse_args(argv)
    should_serve = serve if serve is not None else argv is None

    runtime = bootstrap_mcp_runtime(Path(args.settings))
    if args.dry_run:
        should_serve = False

    if not should_serve:
        print(_format_bootstrap_message(runtime))
        return 0

    _serve_stdio_jsonrpc(runtime, log_level=args.log_level)
    return 0


def main() -> int:
    """Script entrypoint that starts the STDIO MCP server."""

    return run_mcp_server_main()


if __name__ == "__main__":
    raise SystemExit(main())
