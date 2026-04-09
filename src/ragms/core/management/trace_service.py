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
