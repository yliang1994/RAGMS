"""Rule-based semantic metadata generation for smart chunks."""

from __future__ import annotations

import json
from pathlib import Path
import re
from collections import Counter
from typing import Any

_ENGLISH_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
_CJK_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,}")
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[。！？.!?])\s+")
_HEADING_PREFIX_PATTERN = re.compile(r"^#+\s*")
_JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL | re.IGNORECASE)
_LIST_ITEM_PATTERN = re.compile(r"^(?:[-*+]|\d+\.)\s+")

_STOPWORDS = {
    "and",
    "are",
    "for",
    "from",
    "into",
    "that",
    "the",
    "their",
    "this",
    "with",
    "apply",
    "includes",
    "required",
    "using",
}


class MetadataService:
    """Generate deterministic title, summary, and tags for a chunk."""

    def __init__(
        self,
        *,
        short_text_chars: int = 80,
        max_title_chars: int = 60,
        max_summary_chars: int = 180,
        max_tags: int = 5,
    ) -> None:
        self.short_text_chars = short_text_chars
        self.max_title_chars = max_title_chars
        self.max_summary_chars = max_summary_chars
        self.max_tags = max_tags

    def enrich(
        self,
        chunk: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return semantic metadata for the given chunk."""

        content = self._normalize_text(str(chunk.get("content", "")))
        metadata = dict(chunk.get("metadata") or {})
        semantic = dict(metadata.get("semantic") or {})
        path_context = self._path_context(chunk)
        nearby_context = self._nearby_context(metadata, context)

        title = str(
            semantic.get("title")
            or self._derive_title(
                content,
                metadata,
                path_context=path_context,
                nearby_context=nearby_context,
            )
        )
        summary = str(
            semantic.get("summary")
            or self._derive_summary(
                content,
                metadata,
                title=title,
                path_context=path_context,
                nearby_context=nearby_context,
            )
        )
        tags = self._normalize_tags(semantic.get("tags")) or self._derive_tags(
            content,
            metadata,
            title=title,
            path_context=path_context,
            nearby_context=nearby_context,
        )

        return {
            "title": title or "Untitled chunk",
            "summary": summary or f"Summary unavailable for {title or 'this chunk'}.",
            "tags": tags or ["general"],
            "short_text": len(content) <= self.short_text_chars,
            "content_length": len(content),
            "path_context": path_context,
            "nearby_context": nearby_context,
        }

    def _derive_title(
        self,
        content: str,
        metadata: dict[str, Any],
        *,
        path_context: str,
        nearby_context: str,
    ) -> str:
        heading = self._heading_candidate(metadata)
        if heading:
            return self._truncate(heading, self.max_title_chars)

        if nearby_context:
            return self._truncate(nearby_context, self.max_title_chars)

        first_line = self._first_line(content)
        if (
            first_line
            and not _LIST_ITEM_PATTERN.match(first_line)
            and len(first_line) <= self.max_title_chars
            and not first_line.endswith((".", "。"))
        ):
            return first_line

        if path_context and self._list_summary(content):
            return self._truncate(path_context, self.max_title_chars)

        first_sentence = self._first_sentence(content)
        if first_sentence:
            return self._truncate(first_sentence, self.max_title_chars)

        if path_context:
            return self._truncate(path_context, self.max_title_chars)

        return "Untitled chunk"

    def _derive_summary(
        self,
        content: str,
        metadata: dict[str, Any],
        *,
        title: str,
        path_context: str,
        nearby_context: str,
    ) -> str:
        del metadata
        list_summary = self._list_summary(content)
        if list_summary:
            return self._truncate(list_summary, self.max_summary_chars)

        if len(content) <= self.short_text_chars:
            return content

        sentences = [segment.strip() for segment in _SENTENCE_SPLIT_PATTERN.split(content) if segment.strip()]
        if not sentences:
            fallback = nearby_context or path_context or title or content
            return self._truncate(
                fallback or "Summary unavailable for this chunk.",
                self.max_summary_chars,
            )

        summary = " ".join(sentences[:2])
        return self._truncate(summary, self.max_summary_chars)

    def _derive_tags(
        self,
        content: str,
        metadata: dict[str, Any],
        *,
        title: str,
        path_context: str,
        nearby_context: str,
    ) -> list[str]:
        candidates: list[str] = []
        heading = self._heading_candidate(metadata)
        if heading:
            candidates.extend(self._tokenize(heading))

        for section in metadata.get("section_path") or []:
            candidates.extend(self._tokenize(str(section)))

        candidates.extend(self._tokenize(path_context))
        candidates.extend(self._tokenize(nearby_context))
        candidates.extend(self._tokenize(title))
        candidates.extend(self._tokenize(content))
        if not candidates:
            return ["general"]

        counts = Counter(candidates)
        first_seen: dict[str, int] = {}
        for index, token in enumerate(candidates):
            first_seen.setdefault(token, index)

        ranked = sorted(counts, key=lambda token: (-counts[token], first_seen[token], token))
        selected = ranked[: self.max_tags]
        return selected or ["general"]

    @staticmethod
    def _normalize_text(content: str) -> str:
        return "\n".join(line.strip() for line in content.splitlines() if line.strip()).strip()

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        normalized = " ".join(text.split()).strip()
        if len(normalized) <= limit:
            return normalized
        return normalized[: max(limit - 3, 1)].rstrip() + "..."

    @staticmethod
    def _first_line(content: str) -> str:
        for line in content.splitlines():
            candidate = _HEADING_PREFIX_PATTERN.sub("", line.strip())
            if candidate:
                return candidate
        return ""

    @staticmethod
    def _first_sentence(content: str) -> str:
        parts = [segment.strip() for segment in _SENTENCE_SPLIT_PATTERN.split(content) if segment.strip()]
        if parts:
            return parts[0]
        return content.strip()

    @staticmethod
    def _list_summary(content: str) -> str:
        list_items = [
            _LIST_ITEM_PATTERN.sub("", line.strip())
            for line in content.splitlines()
            if _LIST_ITEM_PATTERN.match(line.strip())
        ]
        if not list_items:
            return ""
        return "; ".join(item for item in list_items[:2] if item)

    @staticmethod
    def _heading_candidate(metadata: dict[str, Any]) -> str:
        section_path = metadata.get("section_path") or []
        if section_path:
            last = str(section_path[-1]).strip()
            if last:
                return last
        for key in ("heading", "section_title", "chunk_title"):
            value = str(metadata.get(key) or "").strip()
            if value:
                return value
        return ""

    @staticmethod
    def _normalize_tags(tags: Any) -> list[str]:
        if not isinstance(tags, list):
            return []
        normalized: list[str] = []
        for tag in tags:
            value = str(tag).strip()
            if value and value not in normalized:
                normalized.append(value)
        return normalized

    def _tokenize(self, text: str) -> list[str]:
        english_tokens = [token.lower() for token in _ENGLISH_TOKEN_PATTERN.findall(text)]
        cjk_tokens = _CJK_TOKEN_PATTERN.findall(text)
        merged = english_tokens + cjk_tokens
        return [token for token in merged if token not in _STOPWORDS]

    @staticmethod
    def _path_context(chunk: dict[str, Any]) -> str:
        source_path = str(chunk.get("source_path") or "").strip()
        if not source_path:
            return ""
        stem = Path(source_path).stem.replace("_", " ").replace("-", " ").strip()
        return " ".join(part for part in stem.split() if part)

    @staticmethod
    def _nearby_context(metadata: dict[str, Any], context: dict[str, Any] | None) -> str:
        for key in ("heading", "section_title", "previous_heading", "next_heading"):
            value = str(metadata.get(key) or "").strip()
            if value:
                return value

        if context:
            for key in ("section_title", "heading", "parent_title"):
                value = str(context.get(key) or "").strip()
                if value:
                    return value
        return ""


def extract_json_object(response: str) -> dict[str, Any]:
    """Parse a JSON object from plain or fenced LLM output."""

    candidate = response.strip()
    if not candidate:
        raise ValueError("LLM response is empty")

    fenced_match = _JSON_FENCE_PATTERN.search(candidate)
    if fenced_match:
        candidate = fenced_match.group(1).strip()

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("LLM response does not contain a JSON object") from None
        payload = json.loads(candidate[start : end + 1])

    if not isinstance(payload, dict):
        raise ValueError("LLM response JSON payload must be an object")
    return payload
