from __future__ import annotations

from pathlib import Path


def bootstrap_server() -> dict[str, str]:
    return {
        "service": "ragms-mcp-server",
        "transport": "stdio",
        "status": "bootstrap-ready",
    }


def get_runtime_root() -> Path:
    from ragms import get_project_root

    return get_project_root()

