"""`get_trace_detail` MCP tool adapter."""

from __future__ import annotations

from typing import Any, Callable

from mcp import types

from ragms.core.management import TraceService
from ragms.core.management.trace_service import TraceNotFoundError
from ragms.mcp_server.protocol_handler import JSONRPC_INVALID_PARAMS, ProtocolHandler
from ragms.runtime.container import ServiceContainer


def serialize_trace_detail(trace: dict[str, Any]) -> dict[str, Any]:
    """Normalize a trace detail payload for MCP clients."""

    return {
        "trace_id": trace.get("trace_id"),
        "trace_type": trace.get("trace_type"),
        "status": trace.get("status"),
        "started_at": trace.get("started_at"),
        "finished_at": trace.get("finished_at"),
        "duration_ms": trace.get("duration_ms"),
        "collection": trace.get("collection"),
        "metadata": dict(trace.get("metadata") or {}),
        "error": trace.get("error"),
        "stages": [
            {
                "stage_name": stage.get("stage_name"),
                "status": stage.get("status"),
                "started_at": stage.get("started_at"),
                "finished_at": stage.get("finished_at"),
                "elapsed_ms": stage.get("elapsed_ms"),
                "input_summary": stage.get("input_summary"),
                "output_summary": stage.get("output_summary"),
                "metadata": dict(stage.get("metadata") or {}),
                "error": stage.get("error"),
            }
            for stage in (trace.get("stages") or [])
        ],
        "query": trace.get("query"),
        "top_k_results": list(trace.get("top_k_results") or []),
        "evaluation_metrics": trace.get("evaluation_metrics"),
        "source_path": trace.get("source_path"),
        "document_id": trace.get("document_id"),
        "total_chunks": trace.get("total_chunks"),
        "total_images": trace.get("total_images"),
        "skipped": trace.get("skipped"),
        "run_id": trace.get("run_id"),
        "dataset_version": trace.get("dataset_version"),
        "backends": list(trace.get("backends") or []),
        "metrics_summary": trace.get("metrics_summary"),
        "quality_gate_status": trace.get("quality_gate_status"),
        "baseline_delta": trace.get("baseline_delta"),
    }


def handle_get_trace_detail(
    trace_id: str,
    *,
    runtime: ServiceContainer,
    trace_service: TraceService | None = None,
    protocol_handler: ProtocolHandler | None = None,
) -> types.CallToolResult:
    """Execute the trace-detail tool and return a stable MCP result."""

    handler = protocol_handler or ProtocolHandler()
    request = handler.validate_arguments(
        "get_trace_detail",
        {
            "trace_id": trace_id,
        },
    )
    service = trace_service or TraceService(runtime.settings)

    try:
        payload = service.get_trace_detail(request.trace_id)
    except TraceNotFoundError as exc:
        return handler.build_error_response(
            code=JSONRPC_INVALID_PARAMS,
            message=str(exc),
            data={"trace_id": request.trace_id},
        )
    except Exception as exc:
        error = handler.serialize_exception(exc)
        return handler.build_error_response(
            code=error.code,
            message=error.message,
            data=error.data,
        )

    detail = serialize_trace_detail(payload)
    text = (
        f"Trace {request.trace_id} status={detail['status']} "
        f"type={detail['trace_type']} stages={len(detail['stages'])}."
    )
    return handler.build_success_response(
        text=text,
        structured_content=detail,
    )


def bind_traces_tool(runtime: ServiceContainer) -> Callable[..., types.CallToolResult]:
    """Bind the trace-detail tool to a concrete runtime container."""

    def get_trace_detail(trace_id: str) -> types.CallToolResult:
        return handle_get_trace_detail(
            trace_id=trace_id,
            runtime=runtime,
        )

    get_trace_detail.__name__ = "get_trace_detail"
    get_trace_detail.__doc__ = "Return the full stable trace detail payload."
    return get_trace_detail
