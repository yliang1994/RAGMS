from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _bootstrap_src_path() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    return project_root


PROJECT_ROOT = _bootstrap_src_path()

from ragms import get_project_root
from ragms.mcp_server.server import bootstrap_server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bootstrap the local RAGMS MCP server.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Print bootstrap metadata instead of starting the real server.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    project_root = get_project_root()
    server_info = bootstrap_server()

    if args.check:
        print(f"project_root={project_root}")
        print(f"service={server_info['service']}")
        print(f"transport={server_info['transport']}")
        print(f"status={server_info['status']}")
        return 0

    print("RAGMS MCP server bootstrap is ready.")
    print(f"Project root: {project_root}")
    print("Real MCP server startup will be implemented in later tasks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
