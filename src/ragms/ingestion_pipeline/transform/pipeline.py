"""Transform-stage orchestration for smart chunk enrichment."""

from __future__ import annotations

from typing import Any, Protocol

from ragms.libs.abstractions import BaseTransform


class TransformStep(Protocol):
    """Protocol implemented by transform enrichment steps."""

    def transform(
        self,
        chunks: list[dict[str, Any]],
        *,
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Transform chunks with optional shared context."""


class TransformStepError(RuntimeError):
    """Raised when a transform step fails and fail-open behavior is disabled."""


class TransformPipeline(BaseTransform):
    """Run chunk enrichment steps in a stable, configurable order."""

    def __init__(
        self,
        *,
        smart_chunk_builder: TransformStep | None = None,
        metadata_injector: TransformStep | None = None,
        image_captioner: TransformStep | None = None,
        fail_open: bool = True,
    ) -> None:
        self.smart_chunk_builder = smart_chunk_builder
        self.metadata_injector = metadata_injector
        self.image_captioner = image_captioner
        self.fail_open = fail_open

    def run(
        self,
        chunks: list[dict[str, Any]],
        *,
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute transform steps in order and return enriched smart chunks."""

        state = [dict(chunk) for chunk in chunks]
        shared_context = dict(context or {})
        warnings = list(shared_context.get("transform_warnings") or [])

        for step_name, step in (
            ("smart_chunk_builder", self.smart_chunk_builder),
            ("metadata_injector", self.metadata_injector),
            ("image_captioner", self.image_captioner),
        ):
            if step is None:
                continue
            try:
                state = step.transform(state, context=shared_context)
            except Exception as exc:
                if not self.fail_open:
                    raise TransformStepError(f"Transform step failed: {step_name}") from exc
                warnings.append(f"{step_name}:{exc}")
                shared_context["transform_warnings"] = warnings

        if warnings:
            state = [self._attach_warnings(chunk, warnings) for chunk in state]
        return state

    def transform(
        self,
        chunks: list[dict[str, Any]],
        *,
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Alias BaseTransform contract to the pipeline run method."""

        return self.run(chunks, context=context)

    @staticmethod
    def _attach_warnings(chunk: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
        """Attach pipeline warnings onto chunk metadata without mutating inputs."""

        updated = dict(chunk)
        metadata = dict(updated.get("metadata") or {})
        metadata["transform_warnings"] = list(warnings)
        updated["metadata"] = metadata
        return updated
