from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_bootstrap_entrypoints_can_be_called(project_root: Path) -> None:
    commands = [
        [sys.executable, "scripts/query_cli.py", "--check"],
        [sys.executable, "scripts/run_dashboard.py", "--check"],
        [sys.executable, "scripts/run_mcp_server.py", "--check"],
        [sys.executable, "scripts/ingest_documents.py", "--check"],
    ]

    for command in commands:
        completed = subprocess.run(
            command,
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert "status=bootstrap-ready" in completed.stdout
