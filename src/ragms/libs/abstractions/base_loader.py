from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseLoader(ABC):
    @abstractmethod
    def load(self, source: str, **kwargs: Any) -> dict[str, Any]:
        """Load a source document into a normalized representation."""

