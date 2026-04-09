"""Evaluation orchestration, backend composition, and report persistence."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable
import uuid

from ragms.core.evaluation.dataset_loader import DatasetLoader
from ragms.core.evaluation.report_service import ReportService
from ragms.core.models import EvaluationRunSummary, EvaluationSample, QueryResponsePayload, build_evaluation_input
from ragms.core.trace_collector import TraceManager
from ragms.core.trace_collector.trace_manager import record_evaluation_trace
from ragms.core.trace_collector.trace_utils import serialize_exception

from ragms.observability.metrics import aggregate_evaluation_metrics
from ragms.libs.abstractions import BaseEvaluator
from ragms.libs.factories.evaluator_factory import EvaluatorFactory
from ragms.runtime.settings_models import AppSettings, EvaluationSettings, resolve_evaluation_backends
from ragms.storage.traces import TraceRepository


class CompositeEvaluator(BaseEvaluator):
    """Execute multiple evaluators and normalize their combined result."""

    def __init__(self, evaluators: dict[str, BaseEvaluator]) -> None:
        self.evaluators = dict(evaluators)

    def evaluate(
        self,
        predictions: list[str],
        references: list[str] | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run all configured evaluators and aggregate their backend metrics."""

        backend_results: dict[str, dict[str, Any]] = {}
        sample_errors: list[dict[str, Any]] = []
        aggregate_buckets: dict[str, list[float]] = {}
        for backend_name, evaluator in self.evaluators.items():
            result = evaluator.evaluate(predictions, references, metadata=metadata)
            if "metrics" not in result and "status" not in result:
                result = {
                    "status": "succeeded",
                    "metrics": dict(result),
                    "errors": [],
                }
            backend_results[backend_name] = result
            for error in result.get("errors") or []:
                sample_errors.append(dict(error))
            if result.get("status") != "succeeded":
                continue
            for metric_name, metric_value in (result.get("metrics") or {}).items():
                aggregate_buckets.setdefault(metric_name, []).append(float(metric_value))

        aggregate_metrics = {
            metric_name: sum(values) / len(values)
            for metric_name, values in aggregate_buckets.items()
            if values
        }
        return {
            "aggregate_metrics": aggregate_metrics,
            "backend_results": backend_results,
            "sample_errors": sample_errors,
        }


class EvalRunner:
    """Run dataset-backed evaluations and persist the resulting report and trace."""

    STAGE_ORDER = (
        "dataset_load",
        "sample_build",
        "evaluator_execute",
        "metrics_aggregate",
        "report_persist",
    )

    def __init__(
        self,
        *,
        settings: AppSettings,
        dataset_loader: DatasetLoader | None = None,
        evaluator: CompositeEvaluator | None = None,
        report_service: ReportService | None = None,
        trace_manager: TraceManager | None = None,
        trace_repository: TraceRepository | None = None,
        query_engine: Any | None = None,
        sample_executor: Callable[[EvaluationSample], QueryResponsePayload | dict[str, Any]] | None = None,
    ) -> None:
        self.settings = settings
        self.dataset_loader = dataset_loader or DatasetLoader(settings.paths.data_dir / "evaluation" / "datasets")
        self.evaluator = evaluator or build_evaluator_stack(settings)
        self.report_service = report_service or ReportService(settings)
        self.trace_manager = trace_manager or TraceManager()
        self.trace_repository = trace_repository or TraceRepository(settings.observability.log_file)
        self.query_engine = query_engine
        self.sample_executor = sample_executor

    def run(
        self,
        *,
        dataset_name: str,
        dataset_version: str | None = None,
        labels: list[str] | None = None,
        collection: str | None = None,
        top_k: int = 5,
        baseline_scope: str | None = None,
    ) -> dict[str, Any]:
        """Run one full evaluation and persist its report and evaluation trace."""

        run_id = uuid.uuid4().hex
        trace = self.trace_manager.start_trace(
            "evaluation",
            trace_id=uuid.uuid4().hex,
            collection=collection or self.settings.vector_store.collection,
            metadata={"stage_order": list(self.STAGE_ORDER)},
            run_id=run_id,
            dataset_version=dataset_version,
            backends=resolve_evaluator_backend_set(self.settings),
        )
        try:
            dataset_payload = record_evaluation_trace(
                self.trace_manager,
                trace,
                stage_name="dataset_load",
                input_payload={
                    "dataset_name": dataset_name,
                    "dataset_version": dataset_version,
                    "labels": list(labels or []),
                },
                metadata={"loader": self.dataset_loader.__class__.__name__},
                operation=lambda: self.dataset_loader.load(
                    dataset_name=dataset_name,
                    dataset_version=dataset_version,
                    labels=labels,
                ),
                output_builder=lambda payload: {
                    "sample_count": len(payload.get("samples") or []),
                    "manifest_path": payload.get("manifest_path"),
                },
                metadata_builder=lambda payload: {"collection": payload.get("collection")},
            )
            built_samples, failed_samples = record_evaluation_trace(
                self.trace_manager,
                trace,
                stage_name="sample_build",
                input_payload={
                    "sample_count": len(dataset_payload.get("samples") or []),
                    "collection": collection or dataset_payload.get("collection"),
                    "top_k": top_k,
                },
                metadata={"executor": self._executor_name()},
                operation=lambda: self._build_samples(
                    dataset_payload.get("samples") or [],
                    collection=collection or dataset_payload.get("collection"),
                    top_k=top_k,
                ),
                output_builder=lambda result: {
                    "sample_count": len(result[0]),
                    "failed_count": len(result[1]),
                },
                metadata_builder=lambda result: {"failed_sample_ids": [item["sample_id"] for item in result[1]]},
            )
            evaluated_samples, evaluator_failures = record_evaluation_trace(
                self.trace_manager,
                trace,
                stage_name="evaluator_execute",
                input_payload={
                    "sample_count": len(built_samples),
                    "backend_set": resolve_evaluator_backend_set(self.settings),
                },
                metadata={"evaluator": self.evaluator.__class__.__name__},
                operation=lambda: self._evaluate_samples(built_samples),
                output_builder=lambda result: {
                    "sample_count": len(result[0]),
                    "failed_count": len(result[1]),
                },
                metadata_builder=lambda result: {"failed_sample_ids": [item["sample_id"] for item in result[1]]},
            )
            failed_samples.extend(evaluator_failures)
            aggregate_metrics = record_evaluation_trace(
                self.trace_manager,
                trace,
                stage_name="metrics_aggregate",
                input_payload={
                    "sample_count": len(evaluated_samples),
                    "failed_count": len(failed_samples),
                },
                operation=lambda: aggregate_evaluation_metrics(
                    evaluated_samples,
                    failed_samples=failed_samples,
                ),
            )
            summary = EvaluationRunSummary(
                run_id=run_id,
                trace_id=trace.trace_id,
                collection=str(collection or dataset_payload.get("collection") or self.settings.vector_store.collection),
                dataset_name=dataset_payload.get("dataset_name"),
                dataset_version=dataset_payload.get("dataset_version"),
                backend_set=resolve_evaluator_backend_set(self.settings),
                baseline_scope=baseline_scope,
                config_snapshot={
                    **dict(dataset_payload.get("config_snapshot") or {}),
                    "evaluation": self.settings.evaluation.model_dump(mode="python"),
                },
                started_at=trace.started_at,
                aggregate_metrics=aggregate_metrics,
                quality_gate_status="not_run",
                samples=evaluated_samples,
                failed_samples=failed_samples,
                artifacts={"manifest_path": dataset_payload.get("manifest_path")},
            )
            persisted = record_evaluation_trace(
                self.trace_manager,
                trace,
                stage_name="report_persist",
                input_payload={
                    "run_id": run_id,
                    "sample_count": len(evaluated_samples),
                    "failed_count": len(failed_samples),
                },
                metadata={"service": self.report_service.__class__.__name__},
                operation=lambda: self.report_service.write_report(summary),
                output_builder=lambda payload: {
                    "path": payload.get("path"),
                    "trace_id": payload.get("trace_id"),
                },
            )
        except Exception as exc:
            finished_trace = self.trace_manager.finish_trace(trace, status="failed", error=exc, run_id=run_id)
            self.trace_repository.append(finished_trace)
            raise

        final_status = "failed"
        if evaluated_samples and failed_samples:
            final_status = "partial_success"
        elif evaluated_samples:
            final_status = "succeeded"
        finished_trace = self.trace_manager.finish_trace(
            trace,
            status=final_status,
            run_id=run_id,
            dataset_version=dataset_payload.get("dataset_version"),
            backends=resolve_evaluator_backend_set(self.settings),
            metrics_summary=aggregate_metrics,
            quality_gate_status="not_run",
        )
        self.trace_repository.append(finished_trace)
        return {
            **persisted,
            "run_id": run_id,
            "trace_id": trace.trace_id,
            "status": final_status,
            "aggregate_metrics": aggregate_metrics,
        }

    def _build_samples(
        self,
        samples: list[EvaluationSample],
        *,
        collection: str | None,
        top_k: int,
    ) -> tuple[list[EvaluationSample], list[dict[str, Any]]]:
        built_samples: list[EvaluationSample] = []
        failed_samples: list[dict[str, Any]] = []
        for sample in samples:
            try:
                response = self._execute_sample(sample, collection=collection, top_k=top_k)
                built_samples.append(build_evaluation_input(sample, response))
            except Exception as exc:
                failed_samples.append(
                    {
                        "sample_id": sample.sample_id,
                        "query": sample.query,
                        "stage": "sample_build",
                        "error": serialize_exception(exc),
                    }
                )
        return built_samples, failed_samples

    def _evaluate_samples(
        self,
        samples: list[EvaluationSample],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        evaluated: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        for sample in samples:
            try:
                result = self.evaluator.evaluate(
                    [sample.generated_answer or ""],
                    [sample.ground_truth_answer] if sample.ground_truth_answer else [],
                    metadata={
                        "retrieved_ids": [item.get("chunk_id") for item in sample.retrieved_chunks],
                        "expected_ids": list(sample.expected_chunk_ids),
                        "citations": list(sample.citations),
                        "answer": sample.generated_answer,
                        "evaluation_modes": list(sample.evaluation_modes),
                    },
                )
            except Exception as exc:
                failed.append(
                    {
                        "sample_id": sample.sample_id,
                        "query": sample.query,
                        "stage": "evaluator_execute",
                        "error": serialize_exception(exc),
                    }
                )
                continue

            sample_payload = replace(sample, backend_results=dict(result.get("backend_results") or {})).to_dict()
            sample_payload["metrics_summary"] = dict(result.get("aggregate_metrics") or {})
            evaluated.append(sample_payload)
            for error in result.get("sample_errors") or []:
                failed.append(
                    {
                        "sample_id": sample.sample_id,
                        "query": sample.query,
                        "stage": "evaluator_execute",
                        **dict(error),
                    }
                )
        return evaluated, failed

    def _execute_sample(
        self,
        sample: EvaluationSample,
        *,
        collection: str | None,
        top_k: int,
    ) -> QueryResponsePayload | dict[str, Any]:
        if self.sample_executor is not None:
            return self.sample_executor(sample)
        if self.query_engine is None:
            raise RuntimeError("EvalRunner requires query_engine or sample_executor")
        return self.query_engine.run(
            query=sample.query,
            collection=collection or sample.collection,
            top_k=top_k,
            filters=sample.filters,
        )

    def _executor_name(self) -> str:
        if self.sample_executor is not None:
            return getattr(self.sample_executor, "__name__", self.sample_executor.__class__.__name__)
        if self.query_engine is not None:
            return self.query_engine.__class__.__name__
        return "unconfigured"


def build_evaluator_stack(
    config: AppSettings | EvaluationSettings | dict[str, Any] | None,
) -> CompositeEvaluator:
    """Build a composite evaluator from configured backend order."""

    evaluators = EvaluatorFactory.create_stack(config)
    return CompositeEvaluator(evaluators)


def resolve_evaluator_backend_set(
    config: AppSettings | EvaluationSettings | dict[str, Any] | None,
) -> list[str]:
    """Resolve the configured backend names in execution order."""

    return resolve_evaluation_backends(config)
