"""Repository for persisted evaluation run summaries."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from typing import Any


class EvaluationRepository:
    """Store and query evaluation run summaries in SQLite."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def save_run(
        self,
        payload: dict[str, Any],
        *,
        report_path: str | None = None,
    ) -> dict[str, Any]:
        """Insert or update one evaluation run summary."""

        timestamp = _utc_now()
        stored_report_path = str(report_path or payload.get("artifacts", {}).get("report_path") or payload.get("path") or "")
        self.connection.execute(
            """
            INSERT INTO evaluations (
                run_id,
                trace_id,
                collection,
                dataset_name,
                dataset_version,
                backend_set,
                baseline_scope,
                config_snapshot,
                aggregate_metrics,
                quality_gate_status,
                sample_count,
                failed_sample_count,
                report_path,
                artifacts,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                trace_id = excluded.trace_id,
                collection = excluded.collection,
                dataset_name = excluded.dataset_name,
                dataset_version = excluded.dataset_version,
                backend_set = excluded.backend_set,
                baseline_scope = excluded.baseline_scope,
                config_snapshot = excluded.config_snapshot,
                aggregate_metrics = excluded.aggregate_metrics,
                quality_gate_status = excluded.quality_gate_status,
                sample_count = excluded.sample_count,
                failed_sample_count = excluded.failed_sample_count,
                report_path = excluded.report_path,
                artifacts = excluded.artifacts,
                updated_at = excluded.updated_at
            """,
            (
                str(payload["run_id"]),
                str(payload["trace_id"]),
                str(payload["collection"]),
                payload.get("dataset_name"),
                payload.get("dataset_version"),
                _json_dump(payload.get("backend_set") or []),
                payload.get("baseline_scope"),
                _json_dump(payload.get("config_snapshot") or {}),
                _json_dump(payload.get("aggregate_metrics") or {}),
                payload.get("quality_gate_status"),
                len(payload.get("samples") or []),
                len(payload.get("failed_samples") or []),
                stored_report_path or None,
                _json_dump(payload.get("artifacts") or {}),
                timestamp,
                timestamp,
            ),
        )
        self.connection.commit()
        stored = self.get_run(str(payload["run_id"]))
        if stored is None:  # pragma: no cover
            raise RuntimeError("Failed to persist evaluation run")
        return stored

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Return one evaluation run row by run id."""

        row = self._safe_fetchone(
            """
            SELECT
                run_id,
                trace_id,
                collection,
                dataset_name,
                dataset_version,
                backend_set,
                baseline_scope,
                config_snapshot,
                aggregate_metrics,
                quality_gate_status,
                sample_count,
                failed_sample_count,
                report_path,
                artifacts,
                created_at,
                updated_at
            FROM evaluations
            WHERE run_id = ?
            """,
            (run_id,),
        )
        return None if row is None else self._decode_row(dict(row))

    def list_runs(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        """List persisted evaluation runs ordered by recency."""

        query = """
            SELECT
                run_id,
                trace_id,
                collection,
                dataset_name,
                dataset_version,
                backend_set,
                baseline_scope,
                config_snapshot,
                aggregate_metrics,
                quality_gate_status,
                sample_count,
                failed_sample_count,
                report_path,
                artifacts,
                created_at,
                updated_at
            FROM evaluations
            ORDER BY updated_at DESC, created_at DESC
        """
        if limit is not None:
            query += f" LIMIT {int(limit)}"
        rows = self._safe_fetchall(query, ())
        return [self._decode_row(dict(row)) for row in rows]

    def _safe_fetchone(self, query: str, parameters: tuple[Any, ...]) -> sqlite3.Row | None:
        try:
            return self.connection.execute(query, parameters).fetchone()
        except sqlite3.OperationalError as exc:
            if "no such table: evaluations" in str(exc):
                return None
            raise

    def _safe_fetchall(self, query: str, parameters: tuple[Any, ...]) -> list[sqlite3.Row]:
        try:
            return list(self.connection.execute(query, parameters).fetchall())
        except sqlite3.OperationalError as exc:
            if "no such table: evaluations" in str(exc):
                return []
            raise

    @staticmethod
    def _decode_row(row: dict[str, Any]) -> dict[str, Any]:
        row["backend_set"] = _json_load(row.get("backend_set"), [])
        row["config_snapshot"] = _json_load(row.get("config_snapshot"), {})
        row["aggregate_metrics"] = _json_load(row.get("aggregate_metrics"), {})
        row["artifacts"] = _json_load(row.get("artifacts"), {})
        return row


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_load(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    try:
        return json.loads(str(value))
    except Exception:  # pragma: no cover - defensive boundary
        return default


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
