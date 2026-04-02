"""Minimal Stage A1 entrypoint for the future MCP server."""

from __future__ import annotations

from ragms import get_project_root


def main() -> int:
    """Run the minimal bootstrap entrypoint for Stage A1."""

    project_root = get_project_root()
    print(f"RagMS bootstrap entrypoint ready at: {project_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

