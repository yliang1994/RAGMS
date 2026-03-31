from __future__ import annotations

from pathlib import Path

__all__ = ["__version__", "get_project_root"]

__version__ = "0.1.0"


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]

