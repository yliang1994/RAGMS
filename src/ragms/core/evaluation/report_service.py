"""Read-only evaluation report discovery for dashboard placeholders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ragms.runtime.settings_models import AppSettings


class ReportService:
    """List locally available evaluation runs and report artifacts."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._evaluation_root = settings.paths.data_dir / "evaluation"
        self._runs_dir = self._evaluation_root / "runs"
        self._reports_dir = self._evaluation_root / "reports"

    def list_evaluation_runs(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Return discovered evaluation runs and reports ordered by recency."""

        entries: list[dict[str, Any]] = []
        for path in self._iter_report_files():
            payload = self._load_report_payload(path)
            entries.append(
                {
                    "run_id": payload.get("run_id") or path.stem,
                    "dataset_version": payload.get("dataset_version"),
                    "collection": payload.get("collection"),
                    "metrics_summary": dict(payload.get("metrics_summary") or {}),
                    "quality_gate_status": payload.get("quality_gate_status"),
                    "path": str(path),
                    "updated_at": self._timestamp_for(path),
                }
            )
        entries.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
        if limit is not None:
            return entries[:limit]
        return entries

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

    @staticmethod
    def _timestamp_for(path: Path) -> str:
        return str(path.stat().st_mtime_ns)
