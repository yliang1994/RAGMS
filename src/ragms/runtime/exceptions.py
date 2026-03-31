from __future__ import annotations


class RagmsError(Exception):
    """Base exception for runtime-level failures."""


class DependencyAssemblyError(RagmsError):
    """Raised when the service container fails to assemble dependencies."""

