"""Trace lifecycle manager for query, ingestion, and evaluation flows."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any
import uuid

from ragms.runtime.exceptions import RagMSError

from .stage_recorder import StageRecorder
from .trace_schema import BaseTrace, TRACE_STATUSES, TraceSchemaError, create_trace
from .trace_utils import sanitize_metadata, serialize_exception, utc_now


class TraceLifecycleError(RagMSError):
    """Raised when callers violate trace lifecycle rules."""


@dataclass
class _ActiveStage:
    """Internal mutable stage handle retained until the stage is finished."""

    stage_name: str
    started_at: Any
    input_payload: Any = None
    metadata: dict[str, Any] | None = None


class TraceManager:
    """Manage trace creation, stage transitions, and final convergence."""

    def __init__(self, *, recorder: StageRecorder | None = None) -> None:
        self.recorder = recorder or StageRecorder()

    def start_trace(
        self,
        trace_type: str,
        *,
        trace_id: str | None = None,
        collection: str | None = None,
        metadata: dict[str, Any] | None = None,
        **trace_fields: Any,
    ) -> BaseTrace:
        """Create a new running trace with the requested concrete schema."""

        started_at = utc_now()
        trace = create_trace(
            trace_type,
            trace_id=trace_id or uuid.uuid4().hex,
            started_at=started_at,
            collection=collection,
            metadata=metadata,
            **trace_fields,
        )
        return trace

    def start_stage(
        self,
        trace: BaseTrace,
        stage_name: str,
        *,
        input_payload: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register the start of a stage and reject duplicate stage names."""

        self._ensure_trace_running(trace)
        normalized_name = stage_name.strip()
        if not normalized_name:
            raise TraceLifecycleError("stage_name must not be empty")
        if normalized_name in trace._active_stages or any(stage.stage_name == normalized_name for stage in trace.stages):
            raise TraceLifecycleError(f"duplicate stage name: {normalized_name}")

        trace._active_stages[normalized_name] = _ActiveStage(
            stage_name=normalized_name,
            started_at=utc_now(),
            input_payload=input_payload,
            metadata=dict(metadata or {}),
        )

    def finish_stage(
        self,
        trace: BaseTrace,
        stage_name: str,
        *,
        status: str | None = None,
        output_payload: Any = None,
        metadata: dict[str, Any] | None = None,
        error: BaseException | None = None,
    ) -> None:
        """Finish a previously started stage and append a stable stage trace."""

        resolved_status = self._resolve_status(status, error)
        if resolved_status not in TRACE_STATUSES:
            raise TraceSchemaError(f"unsupported trace status: {resolved_status}")

        active = trace._active_stages.pop(stage_name, None)
        if active is None:
            raise TraceLifecycleError(f"stage is not active: {stage_name}")

        stage_metadata = dict(active.metadata or {})
        stage_metadata.update(dict(metadata or {}))
        trace.stages.append(
            self.recorder.record_stage(
                stage_name=active.stage_name,
                status=resolved_status,
                started_at=active.started_at,
                finished_at=utc_now(),
                input_payload=active.input_payload,
                output_payload=output_payload,
                metadata=stage_metadata,
                error=error,
            )
        )

    def finish_trace(
        self,
        trace: BaseTrace,
        *,
        status: str | None = None,
        error: BaseException | None = None,
        metadata: dict[str, Any] | None = None,
        **trace_fields: Any,
    ) -> BaseTrace:
        """Finish a trace, auto-close unfinished stages, and return the trace."""

        finished_at = utc_now()
        resolved_status = self._resolve_status(status, error)

        while trace._active_stages:
            stage_name, active = next(iter(trace._active_stages.items()))
            del trace._active_stages[stage_name]
            trace.stages.append(
                self.recorder.record_stage(
                    stage_name=active.stage_name,
                    status=resolved_status,
                    started_at=active.started_at,
                    finished_at=finished_at,
                    input_payload=active.input_payload,
                    output_payload=None,
                    metadata=active.metadata,
                    error=error,
                )
            )

        trace.status = resolved_status
        trace.finished_at = trace.finished_at or finished_at.isoformat(timespec="milliseconds").replace("+00:00", "Z")
        trace.duration_ms = max(
            0,
            int(round((finished_at - (trace._started_at_dt or finished_at)).total_seconds() * 1000)),
        )
        trace.error = serialize_exception(error)
        if metadata:
            trace.metadata = sanitize_metadata({**trace.metadata, **metadata})
        self._apply_trace_updates(trace, trace_fields)
        trace._finished_at_dt = finished_at
        return trace

    def _apply_trace_updates(self, trace: BaseTrace, updates: dict[str, Any]) -> None:
        allowed_fields = {field.name for field in fields(trace)}
        for key, value in updates.items():
            if key not in allowed_fields or key.startswith("_"):
                raise TraceLifecycleError(f"unknown trace field: {key}")
            setattr(trace, key, value)

    def _ensure_trace_running(self, trace: BaseTrace) -> None:
        if trace.status != "running":
            raise TraceLifecycleError(f"trace is not running: {trace.trace_id}")

    def _resolve_status(self, status: str | None, error: BaseException | None) -> str:
        if error is not None:
            return "failed" if status is None else status
        return "succeeded" if status is None else status
