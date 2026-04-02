"""Abstract splitter contract for producing stable chunk boundaries."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseSplitter(ABC):
    """Split canonical documents into chunk dictionaries."""

    @abstractmethod
    def split(
        self,
        document: dict[str, Any],
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[dict[str, Any]]:
        """Split a canonical document into chunk records."""

