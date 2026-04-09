"""Repository for persisted evaluation run summaries."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from typing import Any

from ragms.core.models import build_baseline_scope, normalize_backend_set


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
        baseline_scope = str(payload.get("baseline_scope") or "").strip()
        if not baseline_scope:
            baseline_scope = build_baseline_scope(
                collection=str(payload["collection"]),
                dataset_version=payload.get("dataset_version"),
                backend_set=payload.get("backend_set") or [],
            )
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
                baseline_scope,
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

    def get_baseline(self, scope: str) -> dict[str, Any] | None:
        """Return the active baseline row for one scope."""

        for row in self.list_runs():
            if row.get("baseline_scope") != scope:
                continue
            if bool((row.get("artifacts") or {}).get("is_baseline")):
                return row
        return None

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

    def save_baseline_binding(self, scope: str, run_id: str | None) -> dict[str, Any]:
        """Activate one baseline binding for a scope or clear it."""

        rows = self._safe_fetchall(
            """
            SELECT run_id, artifacts
            FROM evaluations
            WHERE baseline_scope = ?
            ORDER BY updated_at DESC, created_at DESC
            """,
            (scope,),
        )
        if run_id is not None and not any(str(row["run_id"]) == str(run_id) for row in rows):
            raise KeyError(f"Unknown baseline run for scope {scope}: {run_id}")

        timestamp = _utc_now()
        for row in rows:
            artifacts = _json_load(row["artifacts"], {})
            artifacts["is_baseline"] = run_id is not None and str(row["run_id"]) == str(run_id)
            self.connection.execute(
                """
                UPDATE evaluations
                SET artifacts = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (
                    _json_dump(artifacts),
                    timestamp,
                    str(row["run_id"]),
                ),
            )
        self.connection.commit()
        return {
            "scope": scope,
            "run_id": run_id,
            "cleared": run_id is None,
        }

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
        row["backend_set"] = normalize_backend_set(row["backend_set"])
        row["config_snapshot"] = _json_load(row.get("config_snapshot"), {})
        row["aggregate_metrics"] = _json_load(row.get("aggregate_metrics"), {})
        row["artifacts"] = _json_load(row.get("artifacts"), {})
        row["is_baseline"] = bool(row["artifacts"].get("is_baseline"))
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
