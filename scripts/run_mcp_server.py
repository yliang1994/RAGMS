"""Local command-line entrypoint for the RagMS MCP server."""

from __future__ import annotations

from collections.abc import Sequence

from ragms.mcp_server.server import run_mcp_server_main as _run_mcp_server_main


def run_mcp_server_main(argv: Sequence[str] | None = None) -> int:
    """Run bootstrap dry-run mode for direct function callers."""

    return _run_mcp_server_main(argv, serve=False)


def main() -> int:
    """Start the STDIO MCP server when invoked as a script."""

    return _run_mcp_server_main()


if __name__ == "__main__":
    raise SystemExit(main())
