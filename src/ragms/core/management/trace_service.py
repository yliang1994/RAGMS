"""Read-only trace query service shared by dashboard and MCP tools."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ragms.runtime.exceptions import RagMSError
from ragms.runtime.settings_models import AppSettings
from ragms.storage.traces import TraceRepository


class TraceNotFoundError(RagMSError):
    """Raised when a requested trace detail cannot be found."""


class TraceService:
    """Provide filtered trace summaries and full trace details."""

    def __init__(
        self,
        settings: AppSettings | None = None,
        *,
        repository: TraceRepository | None = None,
        traces_file: str | Path | None = None,
    ) -> None:
        resolved_repository = repository
        if resolved_repository is None:
            resolved_path = (
                Path(traces_file)
                if traces_file is not None
                else settings.dashboard.traces_file
                if settings is not None
                else Path("logs/traces.jsonl")
            )
            resolved_repository = TraceRepository(resolved_path)
        self.settings = settings
        self.repository = resolved_repository

    def list_traces(
        self,
        *,
        trace_type: str | None = None,
        status: str | None = None,
        collection: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return filtered trace summaries for dashboard and MCP consumers."""

        traces = self.repository.list_traces(
            trace_type=trace_type,
            status=status,
            collection=collection,
            limit=limit,
        )
        return [self.summarize_trace(trace) for trace in traces]

    def get_trace_detail(self, trace_id: str) -> dict[str, Any]:
        """Return the full stable trace payload or raise a domain error."""

        trace = self.repository.get_by_trace_id(trace_id)
        if trace is None:
            raise TraceNotFoundError(f"Trace not found: {trace_id}")
        return trace

    def get_recent_failures(
        self,
        *,
        trace_type: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Return recent failed or partially successful traces in summary form."""

        failed = self.repository.list_traces(
            trace_type=trace_type,
            status="failed",
            limit=limit,
        )
        partial = self.repository.list_traces(
            trace_type=trace_type,
            status="partial_success",
            limit=limit,
        )
        merged = failed + partial
        merged.sort(key=lambda item: str(item.get("started_at") or ""), reverse=True)
        return [self.summarize_trace(trace) for trace in merged[:limit]]

    def compare_traces(self, left_trace_id: str, right_trace_id: str) -> dict[str, Any]:
        """Return a structured comparison between two traces."""

        left = self.get_trace_detail(left_trace_id)
        right = self.get_trace_detail(right_trace_id)

        left_stages = {
            str(stage.get("stage_name")): stage
            for stage in left.get("stages") or []
        }
        right_stages = {
            str(stage.get("stage_name")): stage
            for stage in right.get("stages") or []
        }
        ordered_stage_names = list(dict.fromkeys([*left_stages.keys(), *right_stages.keys()]))
        stage_comparisons = []
        fallback_differences = []
        for stage_name in ordered_stage_names:
            left_stage = left_stages.get(stage_name)
            right_stage = right_stages.get(stage_name)
            comparison = {
                "stage_name": stage_name,
                "left": None if left_stage is None else self._stage_summary(left_stage),
                "right": None if right_stage is None else self._stage_summary(right_stage),
                "elapsed_delta_ms": self._elapsed_delta(left_stage, right_stage),
            }
            stage_comparisons.append(comparison)
            left_fallback = None if left_stage is None else dict(left_stage.get("metadata") or {}).get("fallback_reason")
            right_fallback = None if right_stage is None else dict(right_stage.get("metadata") or {}).get("fallback_reason")
            if left_fallback != right_fallback:
                fallback_differences.append(
                    {
                        "stage_name": stage_name,
                        "left_fallback_reason": left_fallback,
                        "right_fallback_reason": right_fallback,
                    }
                )

        return {
            "left_trace_id": left_trace_id,
            "right_trace_id": right_trace_id,
            "stage_comparisons": stage_comparisons,
            "metric_deltas": {
                "duration_ms": self._numeric_delta(left.get("duration_ms"), right.get("duration_ms")),
                "stage_count": self._numeric_delta(len(left.get("stages") or []), len(right.get("stages") or [])),
            },
            "fallback_differences": fallback_differences,
            "summary": {
                "left_status": left.get("status"),
                "right_status": right.get("status"),
                "left_trace_type": left.get("trace_type"),
                "right_trace_type": right.get("trace_type"),
            },
        }

    def summarize_trace(self, trace: Mapping[str, Any]) -> dict[str, Any]:
        """Build one compact trace summary without losing navigation fields."""

        stages = list(trace.get("stages") or [])
        last_stage = stages[-1] if stages else None
        summary = {
            "trace_id": trace.get("trace_id"),
            "trace_type": trace.get("trace_type"),
            "status": trace.get("status"),
            "collection": trace.get("collection"),
            "started_at": trace.get("started_at"),
            "finished_at": trace.get("finished_at"),
            "duration_ms": trace.get("duration_ms"),
            "stage_count": len(stages),
            "stage_names": [stage.get("stage_name") for stage in stages],
            "last_stage": None if last_stage is None else last_stage.get("stage_name"),
            "error": trace.get("error"),
        }
        for field in ("query", "source_path", "document_id", "total_chunks", "total_images", "skipped"):
            if field in trace:
                summary[field] = trace.get(field)
        return summary

    @staticmethod
    def _stage_summary(stage: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "status": stage.get("status"),
            "elapsed_ms": stage.get("elapsed_ms"),
            "metadata": dict(stage.get("metadata") or {}),
            "error": stage.get("error"),
        }

    @staticmethod
    def _numeric_delta(left: Any, right: Any) -> int | None:
        if left is None or right is None:
            return None
        return int(left) - int(right)

    @staticmethod
    def _elapsed_delta(left_stage: Mapping[str, Any] | None, right_stage: Mapping[str, Any] | None) -> int | None:
        if left_stage is None or right_stage is None:
            return None
        left_elapsed = left_stage.get("elapsed_ms")
        right_elapsed = right_stage.get("elapsed_ms")
        if left_elapsed is None or right_elapsed is None:
            return None
        return int(left_elapsed) - int(right_elapsed)
