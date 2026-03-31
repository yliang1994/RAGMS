from __future__ import annotations

from ragms import get_project_root


def build_dashboard_context() -> dict[str, object]:
    return {
        "title": "RAGMS Local Dashboard",
        "project_root": str(get_project_root()),
        "status": "bootstrap-ready",
    }

