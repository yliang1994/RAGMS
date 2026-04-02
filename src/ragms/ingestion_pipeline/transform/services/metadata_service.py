"""Rule-based semantic metadata generation for smart chunks."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

_ENGLISH_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
_CJK_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,}")
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[。！？.!?])\s+")
_HEADING_PREFIX_PATTERN = re.compile(r"^#+\s*")
_JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL | re.IGNORECASE)

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

        del context
        content = self._normalize_text(str(chunk.get("content", "")))
        metadata = dict(chunk.get("metadata") or {})
        semantic = dict(metadata.get("semantic") or {})

        title = str(semantic.get("title") or self._derive_title(content, metadata))
        summary = str(semantic.get("summary") or self._derive_summary(content))
        tags = self._normalize_tags(semantic.get("tags")) or self._derive_tags(content, metadata)

        return {
            "title": title,
            "summary": summary,
            "tags": tags,
            "short_text": len(content) <= self.short_text_chars,
            "content_length": len(content),
        }

    def _derive_title(self, content: str, metadata: dict[str, Any]) -> str:
        heading = self._heading_candidate(metadata)
        if heading:
            return self._truncate(heading, self.max_title_chars)

        first_line = self._first_line(content)
        if first_line and len(first_line) <= self.max_title_chars and not first_line.endswith((".", "。")):
            return first_line

        first_sentence = self._first_sentence(content)
        return self._truncate(first_sentence or content or "Untitled chunk", self.max_title_chars)

    def _derive_summary(self, content: str) -> str:
        if len(content) <= self.short_text_chars:
            return content

        sentences = [segment.strip() for segment in _SENTENCE_SPLIT_PATTERN.split(content) if segment.strip()]
        if not sentences:
            return self._truncate(content, self.max_summary_chars)

        summary = " ".join(sentences[:2])
        return self._truncate(summary, self.max_summary_chars)

    def _derive_tags(self, content: str, metadata: dict[str, Any]) -> list[str]:
        candidates: list[str] = []
        heading = self._heading_candidate(metadata)
        if heading:
            candidates.extend(self._tokenize(heading))

        for section in metadata.get("section_path") or []:
            candidates.extend(self._tokenize(str(section)))

        candidates.extend(self._tokenize(content))
        if not candidates:
            return []

        counts = Counter(candidates)
        first_seen: dict[str, int] = {}
        for index, token in enumerate(candidates):
            first_seen.setdefault(token, index)

        ranked = sorted(counts, key=lambda token: (-counts[token], first_seen[token], token))
        return ranked[: self.max_tags]

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
