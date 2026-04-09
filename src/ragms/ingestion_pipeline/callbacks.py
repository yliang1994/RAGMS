"""Pipeline callback protocol and event payloads for ingestion orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PipelineEvent:
    """Pipeline-level event emitted when a run begins."""

    trace_id: str
    source_path: str
    document_id: str | None
    total_stages: int
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StageEvent:
    """Stage boundary event emitted before and after each ingestion step."""

    trace_id: str
    source_path: str
    document_id: str | None
    stage: str
    index: int
    total: int
    status: str
    elapsed_ms: float = 0.0
    retry_count: int = 0
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProgressEvent:
    """Coarse-grained progress update for dashboard or logging hooks."""

    trace_id: str
    source_path: str
    document_id: str | None
    completed_stages: int
    total_stages: int
    current_stage: str
    status: str
    elapsed_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ErrorEvent:
    """Structured error event emitted when a stage fails."""

    trace_id: str
    source_path: str
    document_id: str | None
    stage: str
    completed_stages: int
    total_stages: int
    retry_count: int
    error: dict[str, Any] = field(default_factory=dict)


class PipelineCallback:
    """Default no-op callback base used by ingestion pipeline hooks."""

    def on_pipeline_start(self, event: PipelineEvent) -> None:
        """Handle a pipeline start event."""

    def on_stage_start(self, event: StageEvent) -> None:
        """Handle a stage start event."""

    def on_stage_end(self, event: StageEvent) -> None:
        """Handle a stage completion or failure event."""

    def on_progress(self, event: ProgressEvent) -> None:
        """Handle a coarse-grained pipeline progress update."""

    def on_error(self, event: ErrorEvent) -> None:
        """Handle a normalized stage failure event."""
