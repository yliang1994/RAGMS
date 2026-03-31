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

from ragms.runtime.container import build_container


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local RAGMS query CLI bootstrap.")
    parser.add_argument("query", nargs="?", default="health-check")
    parser.add_argument("--check", action="store_true", help="Print bootstrap metadata and exit.")
    return parser


def run_cli() -> int:
    parser = build_parser()
    args = parser.parse_args()
    container = build_container()

    print(f"query={args.query}")
    print(f"default_collection={container.query_engine.config['default_collection']}")
    print(f"llm_provider={container.query_engine.config['llm_provider']}")
    if args.check:
        print("status=bootstrap-ready")
    else:
        print("Query execution pipeline will be implemented in later tasks.")
    return 0


def main() -> int:
    return run_cli()


if __name__ == "__main__":
    raise SystemExit(main())

