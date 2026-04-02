"""Minimal local dashboard bootstrap."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from ragms.runtime.config import load_settings
from ragms.runtime.container import build_container


def run_dashboard(argv: Sequence[str] | None = None) -> int:
    """Load runtime settings and print a placeholder dashboard startup summary."""

    parser = argparse.ArgumentParser(description="Run the local RagMS dashboard bootstrap.")
    parser.add_argument(
        "--settings",
        default="settings.yaml",
        help="Path to the settings.yaml file.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    settings = load_settings(args.settings)
    build_container(settings)
    print(
        "Dashboard bootstrap ready: "
        f"enabled={settings.dashboard.enabled} "
        f"port={settings.dashboard.port} "
        f"traces={settings.dashboard.traces_file}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_dashboard())
