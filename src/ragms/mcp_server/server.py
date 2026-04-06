"""MCP server bootstrap and STDIO runtime entrypoints."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.server.models import InitializationOptions

from ragms.mcp_server.tool_registry import build_tool_registry, register_tools
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
    tool_registry = build_tool_registry()
    register_tools(server, tool_registry)
    setattr(server, "runtime_container", runtime)
    setattr(server, "tool_registry", tool_registry)
    return server


def handle_initialize(server: FastMCP) -> InitializationOptions:
    """Return the initialization payload that the server will expose to clients."""

    return server._mcp_server.create_initialization_options()


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

    _configure_stderr_logging(args.log_level)
    server = create_server(runtime, log_level=args.log_level)
    init_options = handle_initialize(server)
    LOGGER.info(
        "Starting RagMS MCP server name=%s version=%s tools_capability=%s",
        init_options.server_name,
        init_options.server_version,
        bool(init_options.capabilities.tools),
    )
    server.run(transport="stdio")
    return 0


def main() -> int:
    """Script entrypoint that starts the STDIO MCP server."""

    return run_mcp_server_main()


if __name__ == "__main__":
    raise SystemExit(main())
