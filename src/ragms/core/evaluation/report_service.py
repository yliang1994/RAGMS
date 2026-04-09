"""Evaluation report persistence and read APIs for dashboard, CLI, and MCP consumers."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any

from ragms.core.models import EvaluationRunSummary
from ragms.runtime.settings_models import AppSettings
from ragms.storage.sqlite.repositories import EvaluationRepository
from ragms.storage.sqlite.schema import initialize_metadata_schema


class ReportService:
    """Persist and query local evaluation run reports."""

    def __init__(
        self,
        settings: AppSettings,
        *,
        connection: sqlite3.Connection | None = None,
        repository: EvaluationRepository | None = None,
    ) -> None:
        self.settings = settings
        self._evaluation_root = settings.paths.data_dir / "evaluation"
        self._runs_dir = self._evaluation_root / "runs"
        self._reports_dir = self._evaluation_root / "reports"
        self.connection = connection or initialize_metadata_schema(settings.storage.sqlite.path)
        self.repository = repository or EvaluationRepository(self.connection)

    def write_report(self, report: EvaluationRunSummary | dict[str, Any]) -> dict[str, Any]:
        """Persist one evaluation report artifact and its SQLite summary row."""

        payload = report.to_dict() if hasattr(report, "to_dict") else dict(report)
        run_id = str(payload.get("run_id") or "").strip()
        if not run_id:
            raise ValueError("report run_id must not be empty")
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        self._reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = self._reports_dir / f"{run_id}.json"
        run_path = self._runs_dir / f"{run_id}.json"
        payload["artifacts"] = {
            **dict(payload.get("artifacts") or {}),
            "report_path": str(report_path),
            "run_record_path": str(run_path),
        }
        if self.settings.observability.log_file is not None:
            payload["artifacts"].setdefault("trace_file", str(self.settings.observability.log_file))
        serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        report_path.write_text(serialized + "\n", encoding="utf-8")
        run_path.write_text(serialized + "\n", encoding="utf-8")
        stored = self.repository.save_run(payload, report_path=str(report_path))
        return {
            **payload,
            "path": str(report_path),
            "repository_record": stored,
        }

    def list_runs(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Return discovered evaluation runs ordered by recency."""

        entries: dict[str, dict[str, Any]] = {}
        for row in self.repository.list_runs(limit=limit):
            run_id = str(row.get("run_id") or "").strip()
            if not run_id:
                continue
            entries[run_id] = {
                "run_id": run_id,
                "trace_id": row.get("trace_id"),
                "dataset_name": row.get("dataset_name"),
                "dataset_version": row.get("dataset_version"),
                "collection": row.get("collection"),
                "backend_set": list(row.get("backend_set") or []),
                "metrics_summary": dict(row.get("aggregate_metrics") or {}),
                "quality_gate_status": row.get("quality_gate_status"),
                "path": row.get("report_path"),
                "updated_at": row.get("updated_at") or row.get("created_at"),
                "artifacts": dict(row.get("artifacts") or {}),
            }
        for path in self._iter_report_files():
            payload = self._load_report_payload(path)
            run_id = str(payload.get("run_id") or path.stem).strip()
            if not run_id:
                continue
            entries.setdefault(
                run_id,
                {
                    "run_id": run_id,
                    "trace_id": payload.get("trace_id"),
                    "dataset_name": payload.get("dataset_name"),
                    "dataset_version": payload.get("dataset_version"),
                    "collection": payload.get("collection"),
                    "backend_set": list(payload.get("backend_set") or []),
                    "metrics_summary": dict(payload.get("aggregate_metrics") or payload.get("metrics_summary") or {}),
                    "quality_gate_status": payload.get("quality_gate_status"),
                    "path": str(path),
                    "updated_at": self._timestamp_for(path),
                    "artifacts": dict(payload.get("artifacts") or {}),
                },
            )
        ordered = sorted(entries.values(), key=lambda item: str(item.get("updated_at") or ""), reverse=True)
        return ordered[:limit] if limit is not None else ordered

    def list_evaluation_runs(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Compatibility wrapper used by existing dashboard pages."""

        return self.list_runs(limit=limit)

    def load_report_detail(self, run_id: str) -> dict[str, Any] | None:
        """Load one report detail by run id."""

        record = self.repository.get_run(run_id)
        if record is not None:
            payload = self._load_report_from_record(record)
            if payload is not None:
                return self._build_report_detail(payload, path=record.get("report_path"))

        for path in self._iter_report_files():
            payload = self._load_report_payload(path)
            resolved_run_id = payload.get("run_id") or path.stem
            if str(resolved_run_id) != str(run_id):
                continue
            return self._build_report_detail(payload, path=str(path))
        return None

    def _iter_report_files(self) -> list[Path]:
        paths: list[Path] = []
        for directory in (self._runs_dir, self._reports_dir):
            if not directory.is_dir():
                continue
            paths.extend(sorted(directory.rglob("*.json")))
        return paths

    @staticmethod
    def _load_report_payload(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _load_report_from_record(self, record: dict[str, Any]) -> dict[str, Any] | None:
        report_path = record.get("report_path")
        if report_path:
            payload = self._load_report_payload(Path(report_path))
            if payload:
                return payload
        return None

    @staticmethod
    def _build_report_detail(payload: dict[str, Any], *, path: str | None) -> dict[str, Any]:
        return {
            "run_id": payload.get("run_id"),
            "trace_id": payload.get("trace_id"),
            "dataset_name": payload.get("dataset_name"),
            "dataset_version": payload.get("dataset_version"),
            "collection": payload.get("collection"),
            "backend_set": list(payload.get("backend_set") or []),
            "metrics_summary": dict(payload.get("aggregate_metrics") or payload.get("metrics_summary") or {}),
            "quality_gate_status": payload.get("quality_gate_status"),
            "path": path,
            "report": payload,
            "navigation": [
                {"label": "跳转到系统总览", "target_page": "system_overview"},
                {"label": "跳转到数据浏览", "target_page": "data_browser", "collection": payload.get("collection")},
                {"label": "查看评估 Trace", "target_page": "evaluation_panel", "run_id": payload.get("run_id")},
            ],
        }

    @staticmethod
    def _timestamp_for(path: Path) -> str:
        return str(path.stat().st_mtime_ns)
