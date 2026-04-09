"""Evaluation report persistence and read APIs for dashboard, CLI, and MCP consumers."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any

from ragms.core.models import EvaluationRunSummary, build_baseline_scope, normalize_backend_set
from ragms.runtime.settings_models import AppSettings
from ragms.storage.sqlite.repositories import EvaluationRepository
from ragms.storage.sqlite.schema import initialize_metadata_schema


class ReportServiceError(ValueError):
    """Raised when report baseline operations receive invalid scope or run inputs."""


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
        payload["backend_set"] = normalize_backend_set(payload.get("backend_set") or [])
        payload["baseline_scope"] = str(payload.get("baseline_scope") or "").strip() or build_baseline_scope(
            collection=str(payload.get("collection") or ""),
            dataset_version=payload.get("dataset_version"),
            backend_set=payload.get("backend_set") or [],
        )
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

    def set_baseline(
        self,
        run_id: str | None = None,
        *,
        collection: str | None = None,
        dataset_version: str | None = None,
        backend_set: list[str] | None = None,
    ) -> dict[str, Any]:
        """Set or clear the active baseline binding for one canonical scope."""

        if run_id is not None:
            record = self.repository.get_run(run_id)
            if record is None:
                raise ReportServiceError(f"Unknown evaluation run: {run_id}")
            scope = str(record.get("baseline_scope") or "").strip()
            binding = self.repository.save_baseline_binding(scope, run_id)
            baseline = self.repository.get_baseline(scope)
            return {
                **binding,
                "baseline": None if baseline is None else self._summarize_record(baseline),
            }

        scope = self._build_scope(
            collection=collection,
            dataset_version=dataset_version,
            backend_set=backend_set,
        )
        return self.repository.save_baseline_binding(scope, None)

    def get_baseline(
        self,
        *,
        collection: str,
        dataset_version: str,
        backend_set: list[str],
    ) -> dict[str, Any] | None:
        """Return the active baseline summary for one scope."""

        scope = self._build_scope(
            collection=collection,
            dataset_version=dataset_version,
            backend_set=backend_set,
        )
        record = self.repository.get_baseline(scope)
        if record is None:
            return None
        return self._summarize_record(record)

    def compare_runs(self, current_run_id: str, baseline_run_id: str) -> dict[str, Any]:
        """Compare two runs with a stable report contract."""

        current = self._require_report(current_run_id)
        baseline = self._require_report(baseline_run_id)
        current_scope = str(current["report"].get("baseline_scope") or "")
        baseline_scope = str(baseline["report"].get("baseline_scope") or "")
        if current_scope != baseline_scope:
            raise ReportServiceError(
                "Cannot compare runs with different baseline scopes: "
                f"{current_run_id}={current_scope}, {baseline_run_id}={baseline_scope}"
            )
        return {
            "current_run": self._comparison_summary(current),
            "baseline_run": self._comparison_summary(baseline),
            "metric_deltas": self._numeric_deltas(
                current["metrics_summary"],
                baseline["metrics_summary"],
            ),
            "sample_deltas": self._sample_deltas(
                current["report"].get("samples") or [],
                baseline["report"].get("samples") or [],
            ),
            "quality_gate_delta": {
                "current": current.get("quality_gate_status"),
                "baseline": baseline.get("quality_gate_status"),
                "changed": current.get("quality_gate_status") != baseline.get("quality_gate_status"),
            },
            "config_diff_summary": self._config_diff_summary(
                current["report"].get("config_snapshot") or {},
                baseline["report"].get("config_snapshot") or {},
            ),
        }

    def compare_against_baseline(self, run_id: str) -> dict[str, Any] | None:
        """Compare one run against the active baseline in the same scope."""

        current = self._require_report(run_id)
        backend_set = list(current.get("backend_set") or current["report"].get("backend_set") or [])
        dataset_version = str(current.get("dataset_version") or "")
        if not backend_set or not dataset_version:
            return None
        baseline = self.get_baseline(
            collection=str(current["collection"]),
            dataset_version=dataset_version,
            backend_set=backend_set,
        )
        if baseline is None or baseline["run_id"] == run_id:
            return None
        return self.compare_runs(run_id, str(baseline["run_id"]))

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
                "baseline_scope": row.get("baseline_scope"),
                "is_baseline": bool(row.get("is_baseline")),
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
                    "baseline_scope": payload.get("baseline_scope"),
                    "is_baseline": bool((payload.get("artifacts") or {}).get("is_baseline")),
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
            "baseline_scope": payload.get("baseline_scope"),
            "is_baseline": bool((payload.get("artifacts") or {}).get("is_baseline")),
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
    def _numeric_deltas(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
        deltas: dict[str, Any] = {}
        for key in sorted(set(current) | set(baseline)):
            left = current.get(key)
            right = baseline.get(key)
            if isinstance(left, bool) or isinstance(right, bool):
                continue
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                deltas[key] = round(float(left) - float(right), 6)
        return deltas

    @staticmethod
    def _sample_deltas(current_samples: list[dict[str, Any]], baseline_samples: list[dict[str, Any]]) -> dict[str, Any]:
        current_by_id = {str(sample.get("sample_id")): sample for sample in current_samples}
        baseline_by_id = {str(sample.get("sample_id")): sample for sample in baseline_samples}
        shared_ids = sorted(set(current_by_id) & set(baseline_by_id))
        rows = []
        for sample_id in shared_ids:
            rows.append(
                {
                    "sample_id": sample_id,
                    "metric_deltas": ReportService._numeric_deltas(
                        dict(current_by_id[sample_id].get("metrics_summary") or {}),
                        dict(baseline_by_id[sample_id].get("metrics_summary") or {}),
                    ),
                }
            )
        return {
            "shared_sample_count": len(shared_ids),
            "current_only_sample_ids": sorted(set(current_by_id) - set(baseline_by_id)),
            "baseline_only_sample_ids": sorted(set(baseline_by_id) - set(current_by_id)),
            "rows": rows,
        }

    @staticmethod
    def _config_diff_summary(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
        changed_keys = []
        for key in sorted(set(current) | set(baseline)):
            if current.get(key) != baseline.get(key):
                changed_keys.append(key)
        return {
            "changed_keys": changed_keys,
            "changed_count": len(changed_keys),
        }

    @staticmethod
    def _comparison_summary(detail: dict[str, Any]) -> dict[str, Any]:
        return {
            "run_id": detail.get("run_id"),
            "collection": detail.get("collection"),
            "dataset_version": detail.get("dataset_version"),
            "backend_set": list(detail.get("backend_set") or []),
            "baseline_scope": detail.get("baseline_scope"),
            "quality_gate_status": detail.get("quality_gate_status"),
        }

    @staticmethod
    def _summarize_record(record: dict[str, Any]) -> dict[str, Any]:
        return {
            "run_id": record.get("run_id"),
            "trace_id": record.get("trace_id"),
            "collection": record.get("collection"),
            "dataset_name": record.get("dataset_name"),
            "dataset_version": record.get("dataset_version"),
            "backend_set": list(record.get("backend_set") or []),
            "baseline_scope": record.get("baseline_scope"),
            "is_baseline": bool(record.get("is_baseline")),
            "metrics_summary": dict(record.get("aggregate_metrics") or {}),
            "quality_gate_status": record.get("quality_gate_status"),
        }

    @staticmethod
    def _build_scope(
        *,
        collection: str | None,
        dataset_version: str | None,
        backend_set: list[str] | None,
    ) -> str:
        try:
            return build_baseline_scope(
                collection=str(collection or ""),
                dataset_version=dataset_version,
                backend_set=backend_set or [],
            )
        except Exception as exc:  # pragma: no cover - unified service boundary
            raise ReportServiceError(str(exc)) from exc

    def _require_report(self, run_id: str) -> dict[str, Any]:
        detail = self.load_report_detail(run_id)
        if detail is None:
            raise ReportServiceError(f"Unknown evaluation run: {run_id}")
        return detail

    @staticmethod
    def _timestamp_for(path: Path) -> str:
        return str(path.stat().st_mtime_ns)
