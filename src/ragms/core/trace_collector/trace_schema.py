"""Stable trace schema contracts shared by ingestion, query, and evaluation flows."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ragms.runtime.exceptions import RagMSError

from .trace_utils import format_timestamp, sanitize_metadata


TRACE_TYPES = ("query", "ingestion", "evaluation")
TRACE_STATUSES = ("running", "succeeded", "failed", "partial_success", "skipped")


class TraceSchemaError(RagMSError):
    """Raised when a trace or stage payload violates the stable schema contract."""


@dataclass
class StageTrace:
    """Serializable representation of one finished stage within a trace."""

    stage_name: str
    status: str
    started_at: str
    finished_at: str
    elapsed_ms: int
    input_summary: Any = None
    output_summary: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.stage_name = self.stage_name.strip()
        if not self.stage_name:
            raise TraceSchemaError("stage_name must not be empty")
        _validate_status(self.status)
        self.metadata = sanitize_metadata(self.metadata)
        if self.elapsed_ms < 0:
            raise TraceSchemaError("elapsed_ms must not be negative")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the stage into a stable JSON-safe payload."""

        return {
            "stage_name": self.stage_name,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_ms": self.elapsed_ms,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "metadata": dict(self.metadata),
            "error": None if self.error is None else dict(self.error),
        }


@dataclass
class BaseTrace:
    """Common top-level trace structure shared by all trace types."""

    trace_id: str
    trace_type: str
    status: str
    started_at: str
    collection: str | None = None
    finished_at: str | None = None
    duration_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None
    stages: list[StageTrace] = field(default_factory=list)
    _started_at_dt: datetime | None = field(default=None, repr=False)
    _finished_at_dt: datetime | None = field(default=None, repr=False)
    _active_stages: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.trace_id = self.trace_id.strip()
        if not self.trace_id:
            raise TraceSchemaError("trace_id must not be empty")
        _validate_trace_type(self.trace_type)
        _validate_status(self.status)
        self.metadata = sanitize_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the trace into a JSON-safe payload for logs and APIs."""

        payload = {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
            "collection": self.collection,
            "metadata": dict(self.metadata),
            "error": None if self.error is None else dict(self.error),
            "stages": [stage.to_dict() for stage in self.stages],
        }
        payload.update(self._extra_fields())
        return payload

    def _extra_fields(self) -> dict[str, Any]:
        return {}


@dataclass
class QueryTrace(BaseTrace):
    """Top-level query trace contract."""

    query: str | None = None
    top_k_results: list[str] = field(default_factory=list)
    evaluation_metrics: dict[str, Any] | None = None

    def _extra_fields(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "top_k_results": [str(item) for item in self.top_k_results],
            "evaluation_metrics": None if self.evaluation_metrics is None else sanitize_metadata(self.evaluation_metrics),
        }


@dataclass
class IngestionTrace(BaseTrace):
    """Top-level ingestion trace contract."""

    source_path: str | None = None
    document_id: str | None = None
    total_chunks: int | None = None
    total_images: int | None = None
    skipped: bool | str | None = None

    def _extra_fields(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "document_id": self.document_id,
            "total_chunks": self.total_chunks,
            "total_images": self.total_images,
            "skipped": self.skipped,
        }


@dataclass
class EvaluationTrace(BaseTrace):
    """Top-level evaluation trace contract."""

    run_id: str | None = None
    dataset_version: str | None = None
    backends: list[str] = field(default_factory=list)
    metrics_summary: dict[str, Any] | None = None
    quality_gate_status: str | None = None
    baseline_delta: dict[str, Any] | None = None

    def _extra_fields(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dataset_version": self.dataset_version,
            "backends": [str(item) for item in self.backends],
            "metrics_summary": None if self.metrics_summary is None else sanitize_metadata(self.metrics_summary),
            "quality_gate_status": self.quality_gate_status,
            "baseline_delta": None if self.baseline_delta is None else sanitize_metadata(self.baseline_delta),
        }


def create_trace(trace_type: str, *, trace_id: str, started_at: datetime, collection: str | None, metadata: dict[str, Any] | None, **kwargs: Any) -> BaseTrace:
    """Create the concrete trace instance for the requested trace type."""

    trace_classes = {
        "query": QueryTrace,
        "ingestion": IngestionTrace,
        "evaluation": EvaluationTrace,
    }
    _validate_trace_type(trace_type)
    trace_class = trace_classes[trace_type]
    trace = trace_class(
        trace_id=trace_id,
        trace_type=trace_type,
        status="running",
        started_at=format_timestamp(started_at),
        collection=collection,
        metadata=dict(metadata or {}),
        **kwargs,
    )
    trace._started_at_dt = started_at
    return trace


def _validate_trace_type(trace_type: str) -> None:
    if trace_type not in TRACE_TYPES:
        raise TraceSchemaError(f"unsupported trace_type: {trace_type}")


def _validate_status(status: str) -> None:
    if status not in TRACE_STATUSES:
        raise TraceSchemaError(f"unsupported trace status: {status}")
