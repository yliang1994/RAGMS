"""`evaluate_collection` MCP tool adapter."""

from __future__ import annotations

from typing import Any, Callable

from mcp import types

from ragms.core.evaluation import EvalRunner, ReportService
from ragms.core.models import normalize_backend_set
from ragms.core.query_engine import build_query_engine
from ragms.mcp_server.protocol_handler import ProtocolHandler
from ragms.runtime.container import ServiceContainer


def normalize_evaluation_request(
    collection: str,
    dataset: str | None = None,
    metrics: list[str] | None = None,
    eval_options: dict[str, Any] | None = None,
    baseline_mode: str = "compare",
    *,
    protocol_handler: ProtocolHandler | None = None,
) -> dict[str, Any]:
    """Validate and normalize evaluation tool arguments."""

    handler = protocol_handler or ProtocolHandler()
    request = handler.validate_arguments(
        "evaluate_collection",
        {
            "collection": collection,
            "dataset": dataset,
            "metrics": metrics,
            "eval_options": eval_options,
            "baseline_mode": baseline_mode,
        },
    )
    options = dict(request.eval_options or {})
    dataset_name = str(request.dataset or options.pop("dataset_name", "golden")).strip()
    dataset_version = str(options.pop("dataset_version", "v1")).strip()
    if not dataset_name:
        raise ValueError("dataset must not be empty")
    if not dataset_version:
        raise ValueError("dataset_version must not be empty")
    resolved_baseline_mode = str(request.baseline_mode or "compare").strip().lower()
    if resolved_baseline_mode not in {"off", "compare", "set"}:
        raise ValueError("baseline_mode must be one of: off, compare, set")
    backend_set = normalize_backend_set(options.pop("backend_set", None) or options.pop("backends", None) or [])
    return {
        "collection": request.collection,
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "requested_metrics": list(request.metrics or []),
        "backend_set": backend_set,
        "labels": list(options.pop("labels", []) or []),
        "top_k": int(options.pop("top_k", 5) or 5),
        "eval_options": options,
        "baseline_mode": resolved_baseline_mode,
    }


def serialize_evaluation_result(
    result: dict[str, Any],
    *,
    baseline_delta: dict[str, Any] | None = None,
    errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return the stable MCP evaluation result payload."""

    return {
        "run_id": result.get("run_id"),
        "trace_id": result.get("trace_id"),
        "collection": result.get("collection"),
        "dataset_name": result.get("dataset_name"),
        "dataset_version": result.get("dataset_version"),
        "backend_set": list(result.get("backend_set") or []),
        "aggregate_metrics": dict(result.get("aggregate_metrics") or {}),
        "quality_gate_status": result.get("quality_gate_status"),
        "baseline_delta": baseline_delta,
        "failed_samples_count": len(result.get("failed_samples") or []),
        "result_path": result.get("path"),
        "errors": list(errors or []),
    }


def handle_evaluate_collection(
    collection: str,
    dataset: str | None = None,
    metrics: list[str] | None = None,
    eval_options: dict[str, Any] | None = None,
    baseline_mode: str = "compare",
    *,
    runtime: ServiceContainer,
    eval_runner: EvalRunner | None = None,
    report_service: ReportService | None = None,
    protocol_handler: ProtocolHandler | None = None,
) -> types.CallToolResult:
    """Execute evaluation through the shared runner and wrap the result for MCP."""

    handler = protocol_handler or ProtocolHandler()
    try:
        request = normalize_evaluation_request(
            collection=collection,
            dataset=dataset,
            metrics=metrics,
            eval_options=eval_options,
            baseline_mode=baseline_mode,
            protocol_handler=handler,
        )
    except Exception as exc:
        error = handler.serialize_exception(exc)
        return handler.build_error_response(
            code=error.code,
            message=error.message,
            data=error.data,
        )

    service = report_service or runtime.services.get("report_service") or ReportService(runtime.settings)
    runner = eval_runner or runtime.services.get("eval_runner")
    if runner is None:
        runner = EvalRunner(
            settings=runtime.settings,
            report_service=service,
            query_engine=build_query_engine(runtime, settings=runtime.settings),
        )

    try:
        result = runner.run(
            collection=request["collection"],
            dataset_name=request["dataset_name"],
            dataset_version=request["dataset_version"],
            backend_set=request["backend_set"],
            labels=request["labels"],
            top_k=request["top_k"],
        )
        detail = service.load_report_detail(str(result["run_id"])) or dict(result)
        if request["baseline_mode"] == "set":
            service.set_baseline(str(result["run_id"]))
        comparison = None
        if request["baseline_mode"] == "compare":
            comparison = service.compare_against_baseline(str(result["run_id"]))
        payload = serialize_evaluation_result(
            detail.get("report") or detail,
            baseline_delta=None if comparison is None else comparison.get("metric_deltas"),
            errors=[dict(item) for item in ((detail.get("report") or detail).get("failed_samples") or [])],
        )
    except Exception as exc:
        error = handler.serialize_exception(exc)
        return handler.build_error_response(
            code=error.code,
            message=error.message,
            data=error.data,
            structured_content=serialize_evaluation_result(
                {
                    "run_id": None,
                    "trace_id": None,
                    "collection": request["collection"],
                    "dataset_name": request["dataset_name"],
                    "dataset_version": request["dataset_version"],
                    "backend_set": request["backend_set"],
                    "aggregate_metrics": {},
                    "quality_gate_status": "failed",
                    "failed_samples": [],
                    "path": None,
                },
                errors=[{"message": error.message}],
            ),
        )

    text = (
        f"Evaluation run {payload['run_id']} collection={payload['collection']} "
        f"dataset={payload['dataset_name']}/{payload['dataset_version']} "
        f"failed_samples={payload['failed_samples_count']}."
    )
    return handler.build_success_response(
        text=text,
        structured_content=payload,
    )


def bind_evaluation_tool(runtime: ServiceContainer) -> Callable[..., types.CallToolResult]:
    """Bind the evaluation tool to a concrete runtime container."""

    def evaluate_collection(
        collection: str,
        dataset: str | None = None,
        metrics: list[str] | None = None,
        eval_options: dict[str, Any] | None = None,
        baseline_mode: str = "compare",
    ) -> types.CallToolResult:
        return handle_evaluate_collection(
            collection=collection,
            dataset=dataset,
            metrics=metrics,
            eval_options=eval_options,
            baseline_mode=baseline_mode,
            runtime=runtime,
        )

    evaluate_collection.__name__ = "evaluate_collection"
    evaluate_collection.__doc__ = "Run collection evaluation and return structured metrics."
    return evaluate_collection
