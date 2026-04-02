"""Abstract loader contract for converting source files into canonical documents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseLoader(ABC):
    """Load source files into canonical document dictionaries."""

    @abstractmethod
    def load(
        self,
        source_path: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Load a source file and return canonical documents."""

