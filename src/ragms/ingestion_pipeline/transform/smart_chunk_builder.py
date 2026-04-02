"""Rule-based smart chunk refinement for denoising and contextual merging."""

from __future__ import annotations

import json
import hashlib
import re
from typing import Any

from ragms.ingestion_pipeline.transform.services.metadata_service import extract_json_object
from ragms.libs.abstractions import BaseLLM, BaseTransform

_WORDLIKE_PATTERN = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")
_LIST_MARKER_PATTERN = re.compile(r"^(?:[-*+]|\d+\.)\s+")
_TABLE_LINE_PATTERN = re.compile(r"^\|.+\|$")
_TABLE_SEPARATOR_PATTERN = re.compile(r"^\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?$")
_HEADING_ONLY_PATTERN = re.compile(r"^#{1,6}$")
_HEADING_PATTERN = re.compile(r"^#{1,6}\s+\S")
_OCR_SPACED_TEXT_PATTERN = re.compile(r"^(?:[A-Za-z0-9]\s+){3,}[A-Za-z0-9]$")


class SmartChunkBuilder(BaseTransform):
    """Refine coarse chunks with stable denoising and nearby-context merging."""

    _NOISE_LINE_PATTERNS = (
        re.compile(r"^page\s+\d+(?:\s+of\s+\d+)?$", re.IGNORECASE),
        re.compile(r"^\d+\s*/\s*\d+$"),
        re.compile(r"^第\s*\d+\s*页$"),
        re.compile(r"^[\W_]{3,}$"),
    )

    def __init__(
        self,
        *,
        merge_below_chars: int = 100,
        max_merged_chars: int = 1200,
        enable_llm_refine: bool = False,
        llm: BaseLLM | None = None,
        llm_model: str | None = None,
        llm_system_prompt: str = (
            "You are a document chunk refinement engine. "
            "Return strict JSON only with keys content and rationale."
        ),
        llm_prompt_template: str = (
            "Refine the chunk into a self-contained semantic unit while preserving meaning. "
            "Do not invent facts. Respond with JSON only."
        ),
        llm_prompt_version: str = "smart_chunk_refine_v1",
        llm_timeout_seconds: float = 30.0,
        llm_max_retries: int = 1,
    ) -> None:
        self.merge_below_chars = merge_below_chars
        self.max_merged_chars = max_merged_chars
        self.enable_llm_refine = enable_llm_refine
        self._llm = llm
        self.llm_model = llm_model
        self.llm_system_prompt = llm_system_prompt
        self.llm_prompt_template = llm_prompt_template
        self.llm_prompt_version = llm_prompt_version
        self.llm_timeout_seconds = llm_timeout_seconds
        self.llm_max_retries = max(llm_max_retries, 1)

    def transform(
        self,
        chunks: list[dict[str, Any]],
        *,
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Apply denoising before contextual rewrite."""

        refined = self.rewrite(self.denoise(chunks))
        if not self.enable_llm_refine:
            return refined
        return self.refine_with_llm(refined, context=context)

    def denoise(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove obvious noise lines and drop chunks that become empty."""

        cleaned_chunks: list[dict[str, Any]] = []
        for chunk in chunks:
            content = str(chunk.get("content", ""))
            cleaned_content, trace = self._clean_content(content)
            if not cleaned_content:
                continue

            updated = dict(chunk)
            metadata = dict(updated.get("metadata") or {})
            image_refs = list(updated.get("image_refs") or metadata.get("image_refs") or [])
            metadata["image_refs"] = image_refs
            metadata["smart_chunk"] = {
                **dict(metadata.get("smart_chunk") or {}),
                "denoised": bool(trace["actions"]),
                "removed_noise_lines": trace["removed_noise_lines"],
                "rewrite_actions": trace["actions"],
                "original_content_sha256": self._content_digest(content),
                "source_chunk_count": 1,
            }
            if cleaned_content != content:
                metadata["refined_by"] = "rule"

            updated["content"] = cleaned_content
            updated["image_refs"] = image_refs
            updated["metadata"] = metadata
            cleaned_chunks.append(updated)
        return cleaned_chunks

    def rewrite(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge neighboring chunks when they likely belong to the same context."""

        rewritten: list[dict[str, Any]] = []
        index = 0
        while index < len(chunks):
            group = [dict(chunks[index])]
            index += 1
            while index < len(chunks) and self._should_merge(group[-1], chunks[index]):
                projected_length = len(self._join_contents(group + [chunks[index]]))
                if projected_length > self.max_merged_chars:
                    break
                group.append(dict(chunks[index]))
                index += 1
            rewritten.append(self._merge_group(group))
        return rewritten

    def refine_with_llm(
        self,
        chunks: list[dict[str, Any]],
        *,
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Apply optional LLM refinement on top of rule-refined chunk content."""

        llm = self._llm
        refined_chunks: list[dict[str, Any]] = []
        shared_context = dict(context or {})

        for chunk in chunks:
            item = dict(chunk)
            metadata = dict(item.get("metadata") or {})
            smart_chunk = dict(metadata.get("smart_chunk") or {})
            smart_chunk["prompt_version"] = self.llm_prompt_version
            model_name = self.llm_model or getattr(llm, "model", None)
            if model_name:
                smart_chunk["model"] = str(model_name)
            metadata["smart_chunk"] = smart_chunk
            item["metadata"] = metadata

            if llm is None:
                refined_chunks.append(
                    self._attach_llm_fallback(item, reason="llm configuration is unavailable")
                )
                continue

            prompt = self._build_llm_prompt(item, context=shared_context)
            try:
                payload = self._generate_refinement_payload(llm, prompt=prompt)
                content = self._validate_refined_payload(payload)
            except Exception as exc:
                refined_chunks.append(self._attach_llm_fallback(item, reason=str(exc)))
                continue

            item["content"] = content
            metadata = dict(item.get("metadata") or {})
            smart_chunk = dict(metadata.get("smart_chunk") or {})
            smart_chunk["llm_refined"] = True
            smart_chunk["prompt_version"] = self.llm_prompt_version
            if model_name:
                smart_chunk["model"] = str(model_name)
            rationale = str(payload.get("rationale") or "").strip()
            if rationale:
                smart_chunk["llm_rationale"] = rationale
            metadata["smart_chunk"] = smart_chunk
            metadata["refined_by"] = "llm"
            metadata.pop("fallback_reason", None)
            item["metadata"] = metadata
            refined_chunks.append(item)

        return refined_chunks

    @classmethod
    def _is_noise_line(cls, line: str) -> bool:
        if (
            cls._is_markdown_list(line)
            or cls._is_table_line(line)
            or cls._is_code_fence(line)
            or cls._is_heading_marker_only(line)
            or cls._is_heading_text(line)
        ):
            return False
        if any(pattern.match(line) for pattern in cls._NOISE_LINE_PATTERNS):
            return True
        if len(line) < 4:
            return False
        wordlike_chars = len(_WORDLIKE_PATTERN.findall(line))
        return wordlike_chars / max(len(line), 1) < 0.35

    def _should_merge(self, left: dict[str, Any], right: dict[str, Any]) -> bool:
        if left.get("document_id") != right.get("document_id"):
            return False
        if left.get("source_path") != right.get("source_path"):
            return False

        left_index = int(left.get("chunk_index", 0))
        right_index = int(right.get("chunk_index", 0))
        if right_index != left_index + 1:
            return False

        left_content = str(left.get("content", "")).strip()
        right_content = str(right.get("content", "")).strip()
        if not left_content or not right_content:
            return False

        if self._looks_like_standalone_boundary(left_content, right_content):
            return False

        return (
            len(left_content) < self.merge_below_chars
            or len(right_content) < self.merge_below_chars
            or not self._ends_with_terminal(left_content)
            or self._starts_with_continuation(right_content)
        )

    @staticmethod
    def _looks_like_standalone_boundary(left_content: str, right_content: str) -> bool:
        return (
            SmartChunkBuilder._ends_with_terminal(left_content)
            and not SmartChunkBuilder._starts_with_continuation(right_content)
            and len(right_content) >= 60
        )

    @staticmethod
    def _starts_with_continuation(content: str) -> bool:
        return bool(re.match(r"^[a-z0-9,(（\-]", content))

    @staticmethod
    def _ends_with_terminal(content: str) -> bool:
        return bool(re.search(r"[。！？.!?:：]$", content))

    @staticmethod
    def _join_contents(chunks: list[dict[str, Any]]) -> str:
        return "\n\n".join(str(chunk.get("content", "")).strip() for chunk in chunks if chunk.get("content"))

    def _merge_group(self, group: list[dict[str, Any]]) -> dict[str, Any]:
        if len(group) == 1:
            item = dict(group[0])
            metadata = dict(item.get("metadata") or {})
            metadata["image_refs"] = list(item.get("image_refs") or metadata.get("image_refs") or [])
            metadata["smart_chunk"] = {
                **dict(metadata.get("smart_chunk") or {}),
                "merged": False,
                "merged_from": [item.get("chunk_id")],
            }
            item["metadata"] = metadata
            return item

        first = group[0]
        last = group[-1]
        merged_content = self._join_contents(group)
        merged_image_refs = self._merge_image_refs(group)
        metadata = dict(first.get("metadata") or {})
        metadata["image_refs"] = merged_image_refs
        metadata["smart_chunk"] = {
            **dict(metadata.get("smart_chunk") or {}),
            "merged": True,
            "merged_from": [chunk.get("chunk_id") for chunk in group],
            "rewrite_actions": self._merge_rewrite_actions(group) + ["merged_neighbor_context"],
            "original_content_sha256": self._content_digest(self._join_raw_contents(group)),
            "source_chunk_count": len(group),
        }
        metadata["refined_by"] = "rule"

        merged = {
            "chunk_id": first.get("chunk_id"),
            "document_id": first.get("document_id"),
            "content": merged_content,
            "source_path": first.get("source_path"),
            "chunk_index": int(first.get("chunk_index", 0)),
            "start_offset": int(first.get("start_offset", 0)),
            "end_offset": int(last.get("end_offset", int(first.get("end_offset", 0)))),
            "image_refs": merged_image_refs,
            "metadata": metadata,
        }
        return merged

    @staticmethod
    def _merge_image_refs(chunks: list[dict[str, Any]]) -> list[str]:
        merged: list[str] = []
        for chunk in chunks:
            metadata = dict(chunk.get("metadata") or {})
            for image_ref in chunk.get("image_refs") or metadata.get("image_refs") or []:
                if image_ref not in merged:
                    merged.append(str(image_ref))
        return merged

    @staticmethod
    def _join_raw_contents(chunks: list[dict[str, Any]]) -> str:
        return "\n\n".join(str(chunk.get("content", "")) for chunk in chunks)

    @staticmethod
    def _content_digest(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def _merge_rewrite_actions(chunks: list[dict[str, Any]]) -> list[str]:
        merged_actions: list[str] = []
        for chunk in chunks:
            metadata = dict(chunk.get("metadata") or {})
            smart_chunk = dict(metadata.get("smart_chunk") or {})
            for action in smart_chunk.get("rewrite_actions") or []:
                value = str(action)
                if value not in merged_actions:
                    merged_actions.append(value)
        return merged_actions

    def _generate_refinement_payload(self, llm: BaseLLM, *, prompt: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for _ in range(self.llm_max_retries):
            try:
                response = llm.generate(prompt, system_prompt=self.llm_system_prompt)
                return extract_json_object(response)
            except Exception as exc:  # pragma: no cover - exercised through fallback tests
                last_error = exc
        if last_error is None:
            raise ValueError("LLM refinement returned no response")
        raise last_error

    @staticmethod
    def _validate_refined_payload(payload: dict[str, Any]) -> str:
        content = str(payload.get("content") or payload.get("refined_content") or "").strip()
        if not content:
            raise ValueError("LLM refinement returned empty content")
        return content

    def _build_llm_prompt(
        self,
        chunk: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> str:
        metadata = dict(chunk.get("metadata") or {})
        smart_chunk = dict(metadata.get("smart_chunk") or {})
        payload = {
            "instruction": self.llm_prompt_template,
            "prompt_version": self.llm_prompt_version,
            "timeout_seconds": self.llm_timeout_seconds,
            "chunk": {
                "chunk_id": chunk.get("chunk_id"),
                "document_id": chunk.get("document_id"),
                "source_path": chunk.get("source_path"),
                "content": str(chunk.get("content", "")),
                "section_path": list(metadata.get("section_path") or []),
                "rewrite_actions": list(smart_chunk.get("rewrite_actions") or []),
                "merged_from": list(smart_chunk.get("merged_from") or []),
            },
            "context": context or {},
        }
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _attach_llm_fallback(chunk: dict[str, Any], *, reason: str) -> dict[str, Any]:
        updated = dict(chunk)
        metadata = dict(updated.get("metadata") or {})
        smart_chunk = dict(metadata.get("smart_chunk") or {})
        smart_chunk["llm_refined"] = False
        metadata["smart_chunk"] = smart_chunk
        metadata["refined_by"] = str(metadata.get("refined_by") or "rule")
        metadata["fallback_reason"] = reason
        updated["metadata"] = metadata
        return updated

    def _clean_content(self, content: str) -> tuple[str, dict[str, list[str]]]:
        lines = content.splitlines()
        cleaned_lines: list[str] = []
        removed_noise_lines: list[str] = []
        actions: list[str] = []
        in_code_block = False

        for raw_line in lines:
            stripped = raw_line.strip()
            if self._is_code_fence(stripped):
                in_code_block = not in_code_block
                cleaned_lines.append(stripped)
                continue

            if in_code_block:
                cleaned_lines.append(raw_line.rstrip())
                continue

            if not stripped:
                if cleaned_lines and cleaned_lines[-1] != "":
                    cleaned_lines.append("")
                continue

            if self._is_noise_line(stripped):
                removed_noise_lines.append(stripped)
                continue

            normalized_line = self._normalize_line(stripped)
            if normalized_line != stripped:
                self._append_action(actions, "normalized_whitespace")
            if self._looks_like_ocr_spaced_text(normalized_line):
                normalized_line = normalized_line.replace(" ", "")
                self._append_action(actions, "repaired_ocr_spacing")
            cleaned_lines.append(normalized_line)

        repaired_lines = self._repair_broken_blocks(cleaned_lines, actions)
        compact_lines = self._compact_blank_lines(repaired_lines)
        if removed_noise_lines:
            self._append_action(actions, "removed_noise_lines")

        return "\n".join(compact_lines).strip(), {
            "removed_noise_lines": removed_noise_lines,
            "actions": actions,
        }

    def _repair_broken_blocks(self, lines: list[str], actions: list[str]) -> list[str]:
        repaired: list[str] = []
        index = 0
        while index < len(lines):
            current = lines[index]
            if current == "":
                repaired.append("")
                index += 1
                continue

            if repaired and self._is_heading_marker_only(repaired[-1]) and not (
                current == ""
                or self._is_markdown_list(current)
                or self._is_table_line(current)
                or self._is_code_fence(current)
            ):
                repaired[-1] = f"{repaired[-1]} {current}".strip()
                self._append_action(actions, "repaired_heading_break")
                index += 1
                continue

            if repaired and self._should_join_lines(repaired[-1], current):
                previous = repaired[-1]
                repaired[-1] = self._join_lines(previous, current)
                if previous.endswith("-") and current[:1].isalnum():
                    self._append_action(actions, "repaired_hyphen_break")
                else:
                    self._append_action(actions, "repaired_section_break")
                index += 1
                continue

            repaired.append(current)
            index += 1
        return repaired

    @staticmethod
    def _compact_blank_lines(lines: list[str]) -> list[str]:
        compacted: list[str] = []
        for line in lines:
            if line == "" and (not compacted or compacted[-1] == ""):
                continue
            compacted.append(line)
        return compacted

    @staticmethod
    def _normalize_line(line: str) -> str:
        if (
            SmartChunkBuilder._is_markdown_list(line)
            or SmartChunkBuilder._is_table_line(line)
            or SmartChunkBuilder._is_heading_text(line)
            or line.startswith(">")
        ):
            return line
        return " ".join(line.split())

    @staticmethod
    def _append_action(actions: list[str], action: str) -> None:
        if action not in actions:
            actions.append(action)

    @staticmethod
    def _is_markdown_list(line: str) -> bool:
        return bool(_LIST_MARKER_PATTERN.match(line))

    @staticmethod
    def _is_table_line(line: str) -> bool:
        return bool(_TABLE_LINE_PATTERN.match(line) or _TABLE_SEPARATOR_PATTERN.match(line))

    @staticmethod
    def _is_code_fence(line: str) -> bool:
        return line.startswith("```") or line.startswith("~~~")

    @staticmethod
    def _is_heading_marker_only(line: str) -> bool:
        return bool(_HEADING_ONLY_PATTERN.match(line))

    @staticmethod
    def _is_heading_text(line: str) -> bool:
        return bool(_HEADING_PATTERN.match(line))

    @staticmethod
    def _looks_like_ocr_spaced_text(line: str) -> bool:
        return bool(_OCR_SPACED_TEXT_PATTERN.match(line))

    @classmethod
    def _should_join_lines(cls, previous: str, current: str) -> bool:
        if not previous or not current:
            return False
        if previous == "" or current == "":
            return False
        if cls._is_markdown_list(previous) or cls._is_markdown_list(current):
            return False
        if cls._is_table_line(previous) or cls._is_table_line(current):
            return False
        if cls._is_code_fence(previous) or cls._is_code_fence(current):
            return False
        if cls._is_heading_text(previous) or cls._is_heading_text(current):
            return False
        return (
            not cls._ends_with_terminal(previous)
            or cls._starts_with_continuation(current)
            or previous.endswith("-")
        )

    @staticmethod
    def _join_lines(previous: str, current: str) -> str:
        if previous.endswith("-") and current[:1].isalnum():
            return f"{previous[:-1]}{current}"
        return f"{previous.rstrip()} {current.lstrip()}"
