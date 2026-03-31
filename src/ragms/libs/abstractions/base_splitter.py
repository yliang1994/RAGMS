from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseSplitter(ABC):
    @abstractmethod
    def split(self, document: dict[str, Any], **kwargs: Any) -> list[dict[str, Any]]:
        """Split a normalized document into chunks."""

