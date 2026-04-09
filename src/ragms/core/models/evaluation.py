"""Evaluation data contracts used across dataset, query, report, and dashboard flows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ragms.runtime.exceptions import RagMSError

from .response import QueryResponsePayload


class EvaluationModelError(RagMSError):
    """Raised when evaluation sample or report payloads are invalid."""


def normalize_backend_set(backends: list[str] | tuple[str, ...] | None) -> list[str]:
    """Return normalized backend names with order-preserving de-duplication."""

    resolved: list[str] = []
    seen: set[str] = set()
    for backend in backends or []:
        normalized = str(backend).strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        resolved.append(normalized)
    return resolved


def build_baseline_scope(
    *,
    collection: str,
    dataset_version: str | None,
    backend_set: list[str] | tuple[str, ...] | None,
) -> str:
    """Build the canonical baseline scope key."""

    normalized_collection = str(collection).strip()
    normalized_version = str(dataset_version or "").strip()
    if not normalized_collection:
        raise EvaluationModelError("collection must not be empty")
    if not normalized_version:
        raise EvaluationModelError("dataset_version must not be empty")
    normalized_backends = normalize_backend_set(backend_set)
    if not normalized_backends:
        raise EvaluationModelError("backend_set must not be empty")
    return f"{normalized_collection}::{normalized_version}::{'|'.join(normalized_backends)}"


@dataclass(frozen=True)
class EvaluationSample:
    """Normalized one-sample evaluation input and output payload."""

    sample_id: str
    query: str
    collection: str
    dataset_name: str | None = None
    filters: dict[str, Any] = field(default_factory=dict)
    expected_chunk_ids: list[str] = field(default_factory=list)
    expected_sources: list[str] = field(default_factory=list)
    ground_truth_answer: str | None = None
    ground_truth_citations: list[dict[str, Any]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    evaluation_modes: list[str] = field(default_factory=lambda: ["retrieval"])
    dataset_version: str | None = None
    sample_source: str | None = None
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)
    generated_answer: str | None = None
    citations: list[dict[str, Any]] = field(default_factory=list)
    trace_id: str | None = None
    backend_results: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise EvaluationModelError("sample_id must not be empty")
        if not self.query.strip():
            raise EvaluationModelError("query must not be empty")
        if not self.collection.strip():
            raise EvaluationModelError("collection must not be empty")
        modes = [str(item).strip() for item in self.evaluation_modes if str(item).strip()]
        if not modes:
            raise EvaluationModelError("evaluation_modes must contain at least one value")
        invalid = sorted({item for item in modes if item not in {"retrieval", "answer"}})
        if invalid:
            raise EvaluationModelError(f"unsupported evaluation_modes: {', '.join(invalid)}")
        if "answer" in modes and not (self.ground_truth_answer or self.ground_truth_citations):
            raise EvaluationModelError("answer mode requires ground_truth_answer or ground_truth_citations")
        object.__setattr__(self, "evaluation_modes", modes)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe evaluation sample payload."""

        return {
            "sample_id": self.sample_id,
            "query": self.query,
            "collection": self.collection,
            "dataset_name": self.dataset_name,
            "filters": dict(self.filters),
            "expected_chunk_ids": list(self.expected_chunk_ids),
            "expected_sources": list(self.expected_sources),
            "ground_truth_answer": self.ground_truth_answer,
            "ground_truth_citations": [dict(item) for item in self.ground_truth_citations],
            "labels": list(self.labels),
            "evaluation_modes": list(self.evaluation_modes),
            "dataset_version": self.dataset_version,
            "sample_source": self.sample_source,
            "config_snapshot": dict(self.config_snapshot),
            "retrieved_chunks": [dict(item) for item in self.retrieved_chunks],
            "generated_answer": self.generated_answer,
            "citations": [dict(item) for item in self.citations],
            "trace_id": self.trace_id,
            "backend_results": dict(self.backend_results),
        }


@dataclass(frozen=True)
class EvaluationRunSummary:
    """Top-level evaluation report payload."""

    run_id: str
    trace_id: str | None
    collection: str
    dataset_name: str | None
    dataset_version: str | None
    backend_set: list[str] = field(default_factory=list)
    baseline_scope: str | None = None
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    started_at: str | None = None
    finished_at: str | None = None
    aggregate_metrics: dict[str, Any] = field(default_factory=dict)
    quality_gate_status: str | None = None
    samples: list[dict[str, Any]] = field(default_factory=list)
    failed_samples: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise EvaluationModelError("run_id must not be empty")
        if not self.collection.strip():
            raise EvaluationModelError("collection must not be empty")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe report summary payload."""

        baseline_scope = self.baseline_scope
        if not baseline_scope and self.dataset_version and self.backend_set:
            baseline_scope = build_baseline_scope(
                collection=self.collection,
                dataset_version=self.dataset_version,
                backend_set=self.backend_set,
            )
        return {
            "run_id": self.run_id,
            "trace_id": self.trace_id,
            "collection": self.collection,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "backend_set": list(self.backend_set),
            "baseline_scope": baseline_scope,
            "config_snapshot": dict(self.config_snapshot),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "aggregate_metrics": dict(self.aggregate_metrics),
            "quality_gate_status": self.quality_gate_status,
            "samples": [dict(item) for item in self.samples],
            "failed_samples": [dict(item) for item in self.failed_samples],
            "artifacts": dict(self.artifacts),
        }


def normalize_evaluation_sample(
    payload: dict[str, Any],
    *,
    defaults: dict[str, Any] | None = None,
) -> EvaluationSample:
    """Normalize a dataset sample with manifest-level defaults applied."""

    merged = dict(defaults or {})
    merged.update(dict(payload))
    default_filters = dict((defaults or {}).get("filters") or {})
    sample_filters = dict(payload.get("filters") or {})
    merged["filters"] = {**default_filters, **sample_filters}
    merged["labels"] = [str(item).strip() for item in (merged.get("labels") or []) if str(item).strip()]
    merged["config_snapshot"] = dict(merged.get("config_snapshot") or {})
    merged["ground_truth_citations"] = list(merged.get("ground_truth_citations") or [])
    merged["expected_chunk_ids"] = list(merged.get("expected_chunk_ids") or [])
    merged["expected_sources"] = list(merged.get("expected_sources") or [])
    merged["evaluation_modes"] = list(merged.get("evaluation_modes") or ["retrieval"])
    return EvaluationSample(**merged)


def build_evaluation_input(
    sample: EvaluationSample,
    response: QueryResponsePayload | dict[str, Any],
    *,
    backend_results: dict[str, Any] | None = None,
) -> EvaluationSample:
    """Attach query response outputs onto an immutable evaluation sample."""

    payload = response.to_dict() if hasattr(response, "to_dict") else dict(response)
    return EvaluationSample(
        **{
            **sample.to_dict(),
            "retrieved_chunks": list(payload.get("retrieved_chunks") or []),
            "generated_answer": payload.get("answer"),
            "citations": list(payload.get("citations") or []),
            "trace_id": payload.get("trace_id"),
            "backend_results": dict(backend_results or {}),
            "config_snapshot": dict(payload.get("config_snapshot") or sample.config_snapshot),
        }
    )
