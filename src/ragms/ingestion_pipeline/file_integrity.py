"""File integrity helpers for SHA256-based ingestion deduplication."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ragms.storage.sqlite.repositories import IngestionHistoryRepository


class FileIntegrity:
    """Compute file hashes and decide whether an ingestion run can be skipped."""

    def __init__(self, repository: IngestionHistoryRepository) -> None:
        self.repository = repository

    def compute_sha256(self, source_path: str | Path) -> str:
        """Compute the byte-level SHA256 hash for a source file."""

        path = Path(source_path)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def should_skip(self, source_path: str | Path, *, force_rebuild: bool = False) -> bool:
        """Return whether an unchanged source should skip downstream ingestion."""

        if force_rebuild:
            return False
        source_sha256 = self.compute_sha256(source_path)
        return self.repository.get_success_by_sha256(source_sha256) is not None

    def mark_success(
        self,
        source_path: str | Path,
        *,
        source_sha256: str | None = None,
        document_id: str | None = None,
        status: str = "indexed",
        config_version: str | None = None,
    ) -> dict[str, object]:
        """Persist a successful ingestion-history record for a source file."""

        resolved_sha256 = source_sha256 or self.compute_sha256(source_path)
        return self.repository.mark_success(
            source_path=str(source_path),
            source_sha256=resolved_sha256,
            document_id=document_id,
            status=status,
            config_version=config_version,
        )

    def mark_failed(
        self,
        source_path: str | Path,
        *,
        error_message: str,
        source_sha256: str | None = None,
        document_id: str | None = None,
        config_version: str | None = None,
    ) -> dict[str, object]:
        """Persist a failed ingestion-history record for a source file."""

        resolved_sha256 = source_sha256 or self.compute_sha256(source_path)
        return self.repository.mark_failed(
            source_path=str(source_path),
            source_sha256=resolved_sha256,
            document_id=document_id,
            error_message=error_message,
            config_version=config_version,
        )
