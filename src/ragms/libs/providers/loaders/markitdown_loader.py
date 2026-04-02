"""Default loader placeholder aligned with the MarkItDown provider contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ragms.libs.abstractions import BaseLoader


class MarkItDownLoader(BaseLoader):
    """Load source files into a canonical document shape."""

    def __init__(self, *, encoding: str = "utf-8", errors: str = "strict") -> None:
        self.encoding = encoding
        self.errors = errors

    def load(
        self,
        source_path: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Load a file into a single canonical document payload."""

        path = Path(source_path)
        content = path.read_text(encoding=self.encoding, errors=self.errors)
        return [
            {
                "content": content,
                "source_path": str(path),
                "metadata": dict(metadata or {}),
            }
        ]
