"""Shared runtime exception types with unified semantics."""

from __future__ import annotations


class RagMSError(Exception):
    """Base exception for project-specific runtime and assembly failures."""


class RuntimeAssemblyError(RagMSError):
    """Raised when the runtime container cannot assemble requested services."""


class ServiceNotFoundError(RagMSError):
    """Raised when a caller requests an unknown service from the container."""

