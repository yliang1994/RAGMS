from __future__ import annotations

from pathlib import Path
from typing import Any

from ragms.libs.abstractions import BaseLoader


class MarkItDownLoader(BaseLoader):
    SUPPORTED_EXTENSIONS = {
        ".md",
        ".markdown",
        ".txt",
        ".pdf",
        ".doc",
        ".docx",
        ".ppt",
        ".pptx",
        ".xls",
        ".xlsx",
    }

    def __init__(self, *, extract_images: bool = True, output_format: str = "markdown") -> None:
        self.extract_images = extract_images
        self.output_format = output_format

    def load(self, source: str, **kwargs: Any) -> dict[str, Any]:
        if not source or not source.strip():
            raise ValueError("source must not be empty")

        source_path = Path(source).expanduser()
        extension = source_path.suffix.lower()
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"unsupported file type: {extension or '<none>'}")
        if not source_path.exists():
            raise FileNotFoundError(source)

        raw_bytes = source_path.read_bytes()
        content = raw_bytes.decode("utf-8", errors="ignore")

        return {
            "source": str(source_path.resolve()),
            "content": content,
            "metadata": {
                "loader": "markitdown",
                "extract_images": self.extract_images,
                "output_format": self.output_format,
                "source_name": source_path.name,
                "source_type": extension.lstrip("."),
                "size_bytes": len(raw_bytes),
                **kwargs,
            },
        }

