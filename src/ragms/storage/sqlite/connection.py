"""SQLite connection helpers for local metadata persistence."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def create_sqlite_connection(path: str | Path) -> sqlite3.Connection:
    """Create a SQLite connection with local-friendly defaults."""

    resolved_path = Path(path).expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(resolved_path), timeout=30.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection
