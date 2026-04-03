"""Filesystem-backed image persistence for ingestion assets."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


class ImageStorage:
    """Persist image assets into the local `data/images/{collection}` tree."""

    def __init__(self, *, root_dir: str | Path = "data/images") -> None:
        self.root_dir = Path(root_dir).expanduser().resolve()

    def save_image(
        self,
        *,
        image_id: str,
        source_path: str | Path,
        collection: str,
    ) -> dict[str, Any]:
        """Copy one image into managed storage and return the persisted file info."""

        source = Path(source_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Image file does not exist: {source}")

        image_bytes = source.read_bytes()
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        extension = source.suffix or ".bin"
        target_dir = self.root_dir / collection
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{image_id}{extension}"

        existed = target_path.exists()
        if not existed or target_path.read_bytes() != image_bytes:
            temp_path = target_path.with_suffix(f"{extension}.tmp")
            temp_path.write_bytes(image_bytes)
            temp_path.replace(target_path)

        return {
            "image_id": image_id,
            "file_path": str(target_path),
            "image_hash": image_hash,
            "created": not existed,
            "updated": existed and target_path.read_bytes() == image_bytes,
        }
