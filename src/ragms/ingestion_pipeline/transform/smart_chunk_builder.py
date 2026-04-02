"""Rule-based smart chunk refinement for denoising and contextual merging."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from ragms.libs.abstractions import BaseTransform

_WORDLIKE_PATTERN = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")


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
    ) -> None:
        self.merge_below_chars = merge_below_chars
        self.max_merged_chars = max_merged_chars

    def transform(
        self,
        chunks: list[dict[str, Any]],
        *,
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Apply denoising before contextual rewrite."""

        del context
        return self.rewrite(self.denoise(chunks))

    def denoise(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove obvious noise lines and drop chunks that become empty."""

        cleaned_chunks: list[dict[str, Any]] = []
        for chunk in chunks:
            content = str(chunk.get("content", ""))
            kept_lines: list[str] = []
            removed_lines: list[str] = []
            for raw_line in content.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                if self._is_noise_line(line):
                    removed_lines.append(line)
                    continue
                kept_lines.append(line)

            cleaned_content = "\n".join(kept_lines).strip()
            if not cleaned_content:
                continue

            updated = dict(chunk)
            metadata = dict(updated.get("metadata") or {})
            image_refs = list(updated.get("image_refs") or metadata.get("image_refs") or [])
            metadata["image_refs"] = image_refs
            metadata["smart_chunk"] = {
                **dict(metadata.get("smart_chunk") or {}),
                "denoised": bool(removed_lines or cleaned_content != content.strip()),
                "removed_noise_lines": removed_lines,
                "source_chunk_count": 1,
            }

            updated["content"] = cleaned_content
            updated["image_refs"] = image_refs
            updated["metadata"] = metadata
            updated["chunk_id"] = self._build_chunk_id(
                updated,
                content=cleaned_content,
                chunk_index=int(updated.get("chunk_index", 0)),
                start_offset=int(updated.get("start_offset", 0)),
                end_offset=int(updated.get("end_offset", int(updated.get("start_offset", 0)))),
            )
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

    @classmethod
    def _is_noise_line(cls, line: str) -> bool:
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
            "source_chunk_count": len(group),
        }

        merged = {
            "chunk_id": self._build_chunk_id(
                first,
                content=merged_content,
                chunk_index=int(first.get("chunk_index", 0)),
                start_offset=int(first.get("start_offset", 0)),
                end_offset=int(last.get("end_offset", int(first.get("end_offset", 0)))),
            ),
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
    def _build_chunk_id(
        chunk: dict[str, Any],
        *,
        content: str,
        chunk_index: int,
        start_offset: int,
        end_offset: int,
    ) -> str:
        metadata = dict(chunk.get("metadata") or {})
        source_sha256 = metadata.get("source_sha256")
        if source_sha256:
            digest_source = f"{source_sha256}:{chunk_index}:{start_offset}:{end_offset}:{content}"
        else:
            digest_source = (
                f"{chunk.get('document_id', '')}:{chunk.get('source_path', '')}:"
                f"{chunk_index}:{start_offset}:{end_offset}:{content}"
            )
        digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()
        return f"chunk_{digest[:16]}"
