"""SQLite schema bootstrap and migration execution."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

from .connection import create_sqlite_connection


def initialize_metadata_schema(path: str | Path) -> sqlite3.Connection:
    """Create a SQLite connection and apply all pending metadata migrations."""

    connection = create_sqlite_connection(path)
    run_sqlite_migrations(connection)
    return connection


def run_sqlite_migrations(connection: sqlite3.Connection) -> list[str]:
    """Apply bundled SQL migrations exactly once and return applied names."""

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            name TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    applied_names = {
        row["name"] for row in connection.execute("SELECT name FROM schema_migrations").fetchall()
    }

    newly_applied: list[str] = []
    for migration_path in _iter_migration_files():
        if migration_path.name in applied_names:
            continue
        connection.executescript(migration_path.read_text(encoding="utf-8"))
        connection.execute(
            "INSERT INTO schema_migrations (name) VALUES (?)",
            (migration_path.name,),
        )
        newly_applied.append(migration_path.name)

    connection.commit()
    return newly_applied


def _iter_migration_files() -> Iterable[Path]:
    """Yield bundled migration files in lexicographic order."""

    migrations_dir = Path(__file__).with_name("migrations")
    return sorted(migrations_dir.glob("*.sql"))
