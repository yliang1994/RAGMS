"""Minimal local dashboard bootstrap."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import subprocess
import sys

from ragms.observability.dashboard import build_dashboard_context, render_app_shell
from ragms.runtime.config import load_settings
from ragms.runtime.container import build_container


def run_dashboard_main(argv: Sequence[str] | None = None) -> int:
    """Load runtime settings and optionally launch the Streamlit dashboard shell."""

    parser = argparse.ArgumentParser(description="Run the local RagMS dashboard bootstrap.")
    parser.add_argument(
        "--settings",
        default="settings.yaml",
        help="Path to the settings.yaml file.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Launch the Streamlit dashboard server instead of only validating the shell.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    settings = load_settings(args.settings)
    runtime = build_container(settings)
    context = build_dashboard_context(settings, runtime=runtime)
    shell_payload = render_app_shell(context)
    if args.serve:
        return _serve_streamlit_dashboard(args.settings)
    print(
        "Dashboard bootstrap ready: "
        f"enabled={settings.dashboard.enabled} "
        f"port={settings.dashboard.port} "
        f"traces={settings.dashboard.traces_file} "
        f"pages={len(shell_payload['pages'])}"
    )
    return 0


def run_dashboard(argv: Sequence[str] | None = None) -> int:
    """Backward-compatible alias for the dashboard entrypoint."""

    return run_dashboard_main(argv)


def _serve_streamlit_dashboard(settings_path: str) -> int:
    """Launch the real Streamlit process for the dashboard shell."""

    app_path = Path(__file__).resolve().parents[1] / "src" / "ragms" / "observability" / "dashboard" / "app.py"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(load_settings(settings_path).dashboard.port),
        "--",
        "--settings",
        str(settings_path),
    ]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(run_dashboard_main())
