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

    _ANCHOR_FIELDS = (
        "chunk_id",
        "document_id",
        "source_path",
        "chunk_index",
        "start_offset",
        "end_offset",
        "source_ref",
        "image_refs",
    )

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
            previous_state = [dict(chunk) for chunk in state]
            try:
                raw_result = step.transform(state, context=shared_context)
                state = self._normalize_step_result(
                    step_name=step_name,
                    previous_state=previous_state,
                    transformed_state=raw_result,
                )
            except Exception as exc:
                if not self.fail_open:
                    raise TransformStepError(f"Transform step failed: {step_name}") from exc
                warnings.append(f"{step_name}:{exc}")
                shared_context["transform_warnings"] = warnings
                state = [self._attach_step_fallback(chunk, step_name, str(exc)) for chunk in previous_state]

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

    @classmethod
    def _normalize_step_result(
        cls,
        *,
        step_name: str,
        previous_state: list[dict[str, Any]],
        transformed_state: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Restore invariant anchor fields after each step and validate output shape."""

        if len(transformed_state) != len(previous_state):
            raise TransformStepError(
                f"Transform step returned mismatched chunk count: {step_name}"
            )
        return [
            cls._restore_anchor_fields(previous_chunk, transformed_chunk)
            for previous_chunk, transformed_chunk in zip(previous_state, transformed_state, strict=True)
        ]

    @classmethod
    def _restore_anchor_fields(
        cls,
        previous_chunk: dict[str, Any],
        transformed_chunk: dict[str, Any],
    ) -> dict[str, Any]:
        """Keep chunk identity and source anchors stable across transform steps."""

        updated = dict(transformed_chunk)
        previous_metadata = dict(previous_chunk.get("metadata") or {})
        metadata = dict(updated.get("metadata") or {})

        for field in cls._ANCHOR_FIELDS:
            if field == "image_refs":
                image_refs = list(previous_chunk.get("image_refs") or previous_metadata.get("image_refs") or [])
                updated["image_refs"] = image_refs
                metadata["image_refs"] = image_refs
                continue

            previous_value = previous_chunk.get(field)
            if previous_value is not None:
                updated[field] = previous_value
                metadata[field] = previous_value

        updated["metadata"] = metadata
        return updated

    @staticmethod
    def _attach_step_fallback(
        chunk: dict[str, Any],
        step_name: str,
        reason: str,
    ) -> dict[str, Any]:
        """Annotate fail-open fallback information on a preserved chunk."""

        updated = dict(chunk)
        metadata = dict(updated.get("metadata") or {})
        fallbacks = list(metadata.get("transform_fallbacks") or [])
        fallbacks.append({"step": step_name, "reason": reason})
        metadata["transform_fallbacks"] = fallbacks
        updated["metadata"] = metadata
        return updated
