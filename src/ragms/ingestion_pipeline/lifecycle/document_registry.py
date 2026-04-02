"""Document registration and state transition helpers."""

from __future__ import annotations

import hashlib

from ragms.storage.sqlite.repositories import DocumentsRepository


class DocumentRegistryError(ValueError):
    """Raised when a document registration or state transition is invalid."""


_ALLOWED_TRANSITIONS = {
    "pending": {"processing", "skipped", "failed", "deleted"},
    "processing": {"indexed", "failed", "deleted"},
    "indexed": {"processing", "deleted"},
    "skipped": {"processing", "deleted"},
    "failed": {"processing", "deleted"},
    "deleted": set(),
}


class DocumentRegistry:
    """Persist and validate document lifecycle records."""

    def __init__(self, repository: DocumentsRepository) -> None:
        self.repository = repository

    def register(
        self,
        *,
        source_path: str,
        source_sha256: str,
        document_id: str | None = None,
        status: str = "pending",
        current_stage: str = "registered",
    ) -> dict[str, object]:
        """Register or refresh a document record in persistent storage."""

        if status not in _ALLOWED_TRANSITIONS:
            raise DocumentRegistryError(f"Unknown document status: {status}")
        resolved_document_id = document_id or self._build_document_id(source_sha256)
        return self.repository.upsert_document(
            document_id=resolved_document_id,
            source_path=source_path,
            source_sha256=source_sha256,
            status=status,
            current_stage=current_stage,
        )

    def update_status(
        self,
        document_id: str,
        *,
        status: str,
        current_stage: str | None = None,
    ) -> dict[str, object]:
        """Validate and persist a document status transition."""

        existing = self.repository.get_by_document_id(document_id)
        if existing is None:
            raise DocumentRegistryError(f"Unknown document: {document_id}")
        previous_status = str(existing["status"])
        allowed = _ALLOWED_TRANSITIONS.get(previous_status, set())
        if status not in allowed:
            raise DocumentRegistryError(
                f"Illegal document status transition: {previous_status} -> {status}"
            )
        return self.repository.update_status(
            document_id,
            status=status,
            current_stage=current_stage,
        )

    def get(self, document_id: str) -> dict[str, object] | None:
        """Return the current document record by id."""

        return self.repository.get_by_document_id(document_id)

    def find_by_source_path(self, source_path: str) -> dict[str, object] | None:
        """Return the latest document record for a source path."""

        return self.repository.get_by_source_path(source_path)

    @staticmethod
    def _build_document_id(source_sha256: str) -> str:
        """Build a stable default document id from a source hash."""

        digest = hashlib.sha256(source_sha256.encode("utf-8")).hexdigest()
        return f"doc_{digest[:16]}"
