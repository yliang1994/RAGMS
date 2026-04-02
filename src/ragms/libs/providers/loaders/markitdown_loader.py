"""MarkItDown-backed loader for canonical document conversion."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from ragms.libs.abstractions import BaseLoader
from ragms.runtime.exceptions import RagMSError


SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".pdf": "pdf",
    ".docx": "docx",
    ".pptx": "pptx",
    ".xlsx": "xlsx",
}

NATIVE_TEXT_EXTENSIONS = {".md", ".markdown", ".txt"}
IMAGE_PLACEHOLDER_PATTERN = re.compile(r"\[IMAGE:\s*([^\]]+?)\s*\]")


class DocumentLoadError(RagMSError):
    """Raised when a source document cannot be converted into a canonical record."""


class MarkItDownLoader(BaseLoader):
    """Load supported source files into a canonical document shape."""

    def __init__(
        self,
        *,
        encoding: str = "utf-8",
        errors: str = "strict",
        converter: Any | None = None,
    ) -> None:
        self.encoding = encoding
        self.errors = errors
        self._converter = converter

    def load(
        self,
        source_path: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Load a source file into a canonical document payload."""

        path = Path(source_path)
        if not path.is_file():
            raise DocumentLoadError(f"Source file does not exist: {path}")

        suffix = path.suffix.lower()
        doc_type = SUPPORTED_EXTENSIONS.get(suffix)
        if doc_type is None:
            raise DocumentLoadError(f"Unsupported source file type: {suffix or '<no extension>'}")

        raw_bytes = path.read_bytes()
        if not raw_bytes:
            raise DocumentLoadError(f"Source file is empty: {path}")

        if suffix in NATIVE_TEXT_EXTENSIONS:
            content = raw_bytes.decode(self.encoding, errors=self.errors)
            title = path.stem
            convert_result = None
        else:
            content, title, convert_result = self._convert_with_markitdown(path)

        source_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        images = self._normalize_images(
            metadata=dict(metadata or {}),
            convert_result=convert_result,
            content=content,
            source_path=str(path),
        )
        image_occurrences = self._normalize_image_occurrences(
            metadata=dict(metadata or {}),
            convert_result=convert_result,
            content=content,
        )
        canonical_metadata = {
            **dict(metadata or {}),
            "document_id": f"doc_{source_sha256[:16]}",
            "doc_type": doc_type,
            "heading_outline": [],
            "image_occurrences": image_occurrences,
            "images": images,
            "page": 1,
            "source_path": str(path),
            "source_sha256": source_sha256,
            "title": title,
        }
        return [
            {
                "content": content,
                "source_path": str(path),
                "metadata": canonical_metadata,
            }
        ]

    def _convert_with_markitdown(self, path: Path) -> tuple[str, str, Any]:
        """Convert supported binary documents via MarkItDown."""

        converter = self._converter or self._build_converter()
        try:
            result = converter.convert(str(path))
        except Exception as exc:  # pragma: no cover - defensive boundary
            raise DocumentLoadError(f"Failed to convert document with MarkItDown: {path}") from exc

        content = self._extract_text_content(result)
        if not content.strip():
            raise DocumentLoadError(f"Converted document is empty: {path}")

        title = getattr(result, "title", None) or path.stem
        return content, str(title), result

    def _build_converter(self) -> Any:
        """Instantiate the MarkItDown converter lazily."""

        try:
            from markitdown import MarkItDown
        except ImportError as exc:  # pragma: no cover - depends on local environment
            raise DocumentLoadError(
                "markitdown package is required to load non-text documents"
            ) from exc
        return MarkItDown()

    @staticmethod
    def _extract_text_content(result: Any) -> str:
        """Normalize known MarkItDown result shapes into plain text."""

        for attribute in ("text_content", "markdown", "text"):
            value = getattr(result, attribute, None)
            if isinstance(value, str) and value:
                return value
        if isinstance(result, str):
            return result
        return str(result)

    @classmethod
    def _normalize_images(
        cls,
        *,
        metadata: dict[str, Any],
        convert_result: Any | None,
        content: str,
        source_path: str,
    ) -> list[dict[str, Any]]:
        """Normalize image asset metadata into a stable list shape."""

        normalized: dict[str, dict[str, Any]] = {}
        for raw_image in cls._iter_metadata_sequence("images", metadata, convert_result):
            if not isinstance(raw_image, dict):
                continue
            image_id = str(raw_image.get("id") or raw_image.get("image_id") or "").strip()
            if not image_id:
                continue
            normalized[image_id] = {
                "id": image_id,
                "path": str(raw_image.get("path") or ""),
                "page": raw_image.get("page"),
                "position": dict(raw_image.get("position") or {}),
            }

        for occurrence in cls._normalize_image_occurrences(
            metadata=metadata,
            convert_result=convert_result,
            content=content,
        ):
            image_id = str(occurrence["image_id"])
            normalized.setdefault(
                image_id,
                {
                    "id": image_id,
                    "path": "",
                    "page": occurrence.get("page"),
                    "position": dict(occurrence.get("position") or {}),
                },
            )

        return list(normalized.values())

    @classmethod
    def _normalize_image_occurrences(
        cls,
        *,
        metadata: dict[str, Any],
        convert_result: Any | None,
        content: str,
    ) -> list[dict[str, Any]]:
        """Normalize image occurrence metadata or derive it from placeholder positions."""

        raw_occurrences = list(cls._iter_metadata_sequence("image_occurrences", metadata, convert_result))
        normalized: list[dict[str, Any]] = []
        if raw_occurrences:
            for raw_occurrence in raw_occurrences:
                if not isinstance(raw_occurrence, dict):
                    continue
                image_id = str(raw_occurrence.get("image_id") or raw_occurrence.get("id") or "").strip()
                if not image_id:
                    continue
                text_offset = int(raw_occurrence.get("text_offset", 0))
                text_length = int(
                    raw_occurrence.get("text_length")
                    or len(cls._build_image_placeholder(image_id))
                )
                normalized.append(
                    {
                        "image_id": image_id,
                        "text_offset": text_offset,
                        "text_length": text_length,
                        "page": raw_occurrence.get("page"),
                        "position": dict(raw_occurrence.get("position") or {}),
                    }
                )
        else:
            for match in IMAGE_PLACEHOLDER_PATTERN.finditer(content):
                image_id = match.group(1).strip()
                normalized.append(
                    {
                        "image_id": image_id,
                        "text_offset": match.start(),
                        "text_length": len(match.group(0)),
                        "page": None,
                        "position": {},
                    }
                )

        normalized.sort(key=lambda item: (int(item["text_offset"]), str(item["image_id"])))
        return normalized

    @staticmethod
    def _iter_metadata_sequence(
        key: str,
        metadata: dict[str, Any],
        convert_result: Any | None,
    ) -> list[Any]:
        """Collect list-like metadata from request metadata and converter output."""

        values: list[Any] = []
        metadata_value = metadata.get(key)
        if isinstance(metadata_value, list):
            values.extend(metadata_value)
        result_value = getattr(convert_result, key, None)
        if isinstance(result_value, list):
            values.extend(result_value)
        return values

    @staticmethod
    def _build_image_placeholder(image_id: str) -> str:
        """Return the canonical placeholder string for an image id."""

        return f"[IMAGE: {image_id}]"
