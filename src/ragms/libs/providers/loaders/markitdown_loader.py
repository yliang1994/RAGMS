"""MarkItDown-backed loader for canonical document conversion."""

from __future__ import annotations

import hashlib
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
        else:
            content, title = self._convert_with_markitdown(path)

        source_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        canonical_metadata = {
            **dict(metadata or {}),
            "document_id": f"doc_{source_sha256[:16]}",
            "doc_type": doc_type,
            "heading_outline": [],
            "images": [],
            "page": 1,
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

    def _convert_with_markitdown(self, path: Path) -> tuple[str, str]:
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
        return content, str(title)

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
