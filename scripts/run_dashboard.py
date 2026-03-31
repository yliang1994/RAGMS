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

from ragms.observability.dashboard.app import build_dashboard_context


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local RAGMS dashboard bootstrap.")
    parser.add_argument("--check", action="store_true", help="Print bootstrap metadata and exit.")
    return parser


def run_dashboard() -> int:
    parser = build_parser()
    args = parser.parse_args()
    context = build_dashboard_context()

    print(f"title={context['title']}")
    print(f"project_root={context['project_root']}")
    if args.check:
        print(f"status={context['status']}")
    else:
        print("Dashboard UI will be implemented in later tasks.")
    return 0


def main() -> int:
    return run_dashboard()


if __name__ == "__main__":
    raise SystemExit(main())

