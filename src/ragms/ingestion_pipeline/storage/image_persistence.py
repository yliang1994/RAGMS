"""Storage writer that persists image assets and image-index mappings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ragms.storage.images import ImageStorage
from ragms.storage.sqlite.repositories import ImagesRepository


class ImageStorageWriter:
    """Persist chunk-linked images to disk and SQLite in one coordinated step."""

    def __init__(
        self,
        *,
        image_storage: ImageStorage,
        repository: ImagesRepository,
        collection: str = "default",
    ) -> None:
        self.image_storage = image_storage
        self.repository = repository
        self.collection = collection

    def save_all(self, records: list[Any]) -> dict[str, Any]:
        """Persist all image assets referenced by the provided chunk records."""

        stored_rows: list[dict[str, Any]] = []
        persisted_paths: set[str] = set()
        for record in records:
            images = {
                str(image.get("id")): dict(image)
                for image in (record.metadata.get("images") or [])
                if isinstance(image, dict) and image.get("id")
            }
            for image_id in list(record.image_refs or []):
                image = images.get(str(image_id))
                if image is None:
                    continue
                stored = self.image_storage.save_image(
                    image_id=str(image["id"]),
                    source_path=str(image.get("path") or ""),
                    collection=self.collection,
                )
                persisted_paths.add(str(stored["file_path"]))
                stored_rows.append(
                    self.repository.upsert_image(
                        image_id=str(image["id"]),
                        document_id=str(record.document_id),
                        chunk_id=str(record.chunk_id),
                        file_path=str(stored["file_path"]),
                        source_path=str(Path(image.get("path") or "")),
                        image_hash=str(stored["image_hash"]),
                        page=int(image.get("page")) if image.get("page") is not None else None,
                        position=dict(image.get("position") or {}),
                    )
                )
        return {
            "stored_images": stored_rows,
            "stored_count": len(stored_rows),
            "file_count": len(persisted_paths),
        }
