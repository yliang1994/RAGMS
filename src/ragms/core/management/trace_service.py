"""Read-only trace query service shared by dashboard and MCP tools."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ragms.runtime.exceptions import RagMSError
from ragms.runtime.settings_models import AppSettings
from ragms.storage.traces import TraceRepository

INGESTION_STAGE_ALIASES = {
    "split": "chunking",
    "embed": "embedding",
    "upsert": "storage",
}


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
        trace_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return filtered trace summaries for dashboard and MCP consumers."""

        traces = self.repository.list_traces(
            trace_type=trace_type,
            status=status,
            collection=collection,
            limit=limit,
        )
        summaries = [self.summarize_trace(trace) for trace in traces]
        if not trace_id:
            return summaries

        needle = str(trace_id).strip().lower()
        return [
            summary
            for summary in summaries
            if needle in str(summary.get("trace_id") or "").lower()
        ]

    def get_trace_detail(self, trace_id: str) -> dict[str, Any]:
        """Return the full stable trace payload or raise a domain error."""

        trace = self.repository.get_by_trace_id(trace_id)
        if trace is None:
            raise TraceNotFoundError(f"Trace not found: {trace_id}")
        summary = self.summarize_trace(trace)
        return {
            **trace,
            "summary": summary,
            "normalized_stage_names": [
                self._normalize_stage_name(trace.get("trace_type"), stage.get("stage_name"))
                for stage in trace.get("stages") or []
            ],
            "navigation": self._build_navigation(trace),
        }

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
        merged_by_id: dict[str, dict[str, Any]] = {}
        for trace in failed + partial:
            trace_id = str(trace.get("trace_id") or f"trace-{len(merged_by_id)}")
            merged_by_id[trace_id] = trace
        merged = sorted(
            merged_by_id.values(),
            key=lambda item: str(item.get("finished_at") or item.get("started_at") or ""),
            reverse=True,
        )
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
                "top_k_results": self._numeric_delta(
                    len(left.get("top_k_results") or []),
                    len(right.get("top_k_results") or []),
                ),
                "citation_count": self._query_metric_delta(left, right, stage_name="response_build", metadata_key="citation_count"),
                "retrieved_count": self._query_metric_delta(left, right, stage_name="dense_retrieval", metadata_key="retrieved_count"),
            },
            "fallback_differences": fallback_differences,
            "query_differences": {
                "retrieval": self._query_metric_comparison(left, right, stage_name="dense_retrieval", metadata_key="retrieved_count"),
                "response_build": self._query_metric_comparison(left, right, stage_name="response_build", metadata_key="citation_count"),
                "rerank": self._query_metric_comparison(left, right, stage_name="rerank", metadata_key="backend"),
                "answer_generation": self._query_metric_comparison(left, right, stage_name="answer_generation", metadata_key="provider"),
            },
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
            "stage_names": [
                self._normalize_stage_name(trace.get("trace_type"), stage.get("stage_name"))
                for stage in stages
            ],
            "last_stage": None
            if last_stage is None
            else self._normalize_stage_name(trace.get("trace_type"), last_stage.get("stage_name")),
            "error": trace.get("error"),
            "target_page": self._target_page_for_trace_type(trace.get("trace_type")),
        }
        for field in ("query", "source_path", "document_id", "total_chunks", "total_images", "skipped"):
            if field in trace:
                summary[field] = trace.get(field)
        for field in ("run_id", "dataset_version", "backends", "metrics_summary", "quality_gate_status"):
            if field in trace:
                summary[field] = trace.get(field)
        return summary

    @staticmethod
    def _normalize_stage_name(trace_type: Any, stage_name: Any) -> str | None:
        normalized = str(stage_name or "").strip()
        if not normalized:
            return None
        if str(trace_type or "").strip().lower() == "ingestion":
            return INGESTION_STAGE_ALIASES.get(normalized, normalized)
        return normalized

    @classmethod
    def _build_navigation(cls, trace: Mapping[str, Any]) -> list[dict[str, Any]]:
        navigation = [
            {
                "label": "查看同类 Trace 列表",
                "target_page": cls._target_page_for_trace_type(trace.get("trace_type")),
                "trace_id": trace.get("trace_id"),
            }
        ]
        document_id = trace.get("document_id")
        source_path = trace.get("source_path")
        if document_id is not None:
            navigation.append(
                {
                    "label": "跳转到数据浏览",
                    "target_page": "data_browser",
                    "document_id": document_id,
                }
            )
        if source_path is not None:
            navigation.append(
                {
                    "label": "查看源文件",
                    "target_page": "data_browser",
                    "source_path": source_path,
                }
            )
        return navigation

    @staticmethod
    def _target_page_for_trace_type(trace_type: Any) -> str:
        if str(trace_type or "").strip().lower() == "ingestion":
            return "ingestion_trace"
        if str(trace_type or "").strip().lower() == "evaluation":
            return "evaluation_panel"
        return "query_trace"

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

    @staticmethod
    def _stage_by_name(trace: Mapping[str, Any], stage_name: str) -> Mapping[str, Any] | None:
        for stage in trace.get("stages") or []:
            if str(stage.get("stage_name") or "") == stage_name:
                return stage
        return None

    @classmethod
    def _query_metric_delta(
        cls,
        left: Mapping[str, Any],
        right: Mapping[str, Any],
        *,
        stage_name: str,
        metadata_key: str,
    ) -> int | None:
        left_stage = cls._stage_by_name(left, stage_name)
        right_stage = cls._stage_by_name(right, stage_name)
        left_value = None if left_stage is None else dict(left_stage.get("metadata") or {}).get(metadata_key)
        right_value = None if right_stage is None else dict(right_stage.get("metadata") or {}).get(metadata_key)
        if left_value is None or right_value is None:
            return None
        try:
            return int(left_value) - int(right_value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _query_metric_comparison(
        cls,
        left: Mapping[str, Any],
        right: Mapping[str, Any],
        *,
        stage_name: str,
        metadata_key: str,
    ) -> dict[str, Any]:
        left_stage = cls._stage_by_name(left, stage_name)
        right_stage = cls._stage_by_name(right, stage_name)
        return {
            "stage_name": stage_name,
            "metric": metadata_key,
            "left": None if left_stage is None else dict(left_stage.get("metadata") or {}).get(metadata_key),
            "right": None if right_stage is None else dict(right_stage.get("metadata") or {}).get(metadata_key),
        }
