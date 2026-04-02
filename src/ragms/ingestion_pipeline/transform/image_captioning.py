"""Image caption generation and chunk injection for multimodal transform."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ragms.libs.abstractions import BaseTransform, BaseVisionLLM
from ragms.storage.sqlite.repositories import ProcessingCacheRepository

IMAGE_CAPTION_PROMPT = "Describe the image faithfully for retrieval and cite visible signals."


def inject_image_caption(
    chunks: list[dict[str, Any]],
    *,
    vision_llm: BaseVisionLLM | None = None,
    cache_repository: ProcessingCacheRepository | None = None,
    model: str | None = None,
    prompt: str = IMAGE_CAPTION_PROMPT,
    prompt_version: str = "image_caption_v1",
) -> list[dict[str, Any]]:
    """Generate and inject image captions into chunk content and metadata."""

    injector = ImageCaptionInjector(
        vision_llm=vision_llm,
        cache_repository=cache_repository,
        model=model,
        prompt=prompt,
        prompt_version=prompt_version,
    )
    return injector.transform(chunks)


def generate_image_caption(
    image_path: str | Path,
    *,
    vision_llm: BaseVisionLLM,
    cache_repository: ProcessingCacheRepository | None = None,
    model: str | None = None,
    prompt: str = IMAGE_CAPTION_PROMPT,
    prompt_version: str = "image_caption_v1",
    context: str | None = None,
) -> dict[str, Any]:
    """Generate a caption for one image, reusing cache when available."""

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    image_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    resolved_model = str(model or getattr(vision_llm, "model", None) or "unknown")

    if cache_repository is not None:
        cached = cache_repository.get_caption(
            image_hash=image_hash,
            model=resolved_model,
            prompt_version=prompt_version,
        )
        if cached is not None:
            return {
                "caption": str(cached["payload"]),
                "image_hash": image_hash,
                "model": resolved_model,
                "prompt_version": prompt_version,
                "cached": True,
            }

    caption = vision_llm.caption(path, prompt=prompt, context=context)
    if not str(caption).strip():
        raise ValueError(f"Vision LLM returned an empty caption for {path}")

    if cache_repository is not None:
        cache_repository.upsert_caption(
            image_hash=image_hash,
            model=resolved_model,
            prompt_version=prompt_version,
            image_path=str(path),
            caption=str(caption).strip(),
        )

    return {
        "caption": str(caption).strip(),
        "image_hash": image_hash,
        "model": resolved_model,
        "prompt_version": prompt_version,
        "cached": False,
    }


class ImageCaptionInjector(BaseTransform):
    """Attach image captions onto chunks using a vision-language model."""

    def __init__(
        self,
        *,
        vision_llm: BaseVisionLLM | None = None,
        cache_repository: ProcessingCacheRepository | None = None,
        model: str | None = None,
        prompt: str = IMAGE_CAPTION_PROMPT,
        prompt_version: str = "image_caption_v1",
    ) -> None:
        self.vision_llm = vision_llm
        self.cache_repository = cache_repository
        self.model = model
        self.prompt = prompt
        self.prompt_version = prompt_version

    def transform(
        self,
        chunks: list[dict[str, Any]],
        *,
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Inject image captions into all image-bearing chunks."""

        del context
        transformed: list[dict[str, Any]] = []
        for chunk in chunks:
            transformed.append(self._transform_chunk(chunk))
        return transformed

    def _transform_chunk(self, chunk: dict[str, Any]) -> dict[str, Any]:
        item = dict(chunk)
        metadata = dict(item.get("metadata") or {})
        image_refs = list(item.get("image_refs") or metadata.get("image_refs") or [])
        occurrences = [
            dict(occurrence)
            for occurrence in metadata.get("image_occurrences") or []
            if isinstance(occurrence, dict)
        ]
        if not image_refs and not occurrences:
            return item

        images_by_id = {
            str(image.get("id")): dict(image)
            for image in metadata.get("images") or []
            if isinstance(image, dict) and image.get("id")
        }
        relevant_image_ids = self._collect_relevant_image_ids(image_refs, occurrences)
        caption_entries: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []

        for image_id in relevant_image_ids:
            image = images_by_id.get(image_id, {"id": image_id, "path": ""})
            try:
                entry = self._caption_image(item, image)
            except Exception as exc:
                failures.append({"image_id": image_id, "reason": str(exc)})
                continue
            caption_entries.append(entry)

        updated_content = str(item.get("content", ""))
        if caption_entries and occurrences:
            updated_content = self._inject_occurrence_captions(
                updated_content,
                occurrences=occurrences,
                captions_by_id={entry["image_id"]: entry["caption"] for entry in caption_entries},
            )
        elif caption_entries:
            appendix = "\n".join(
                f"- {entry['image_id']}: {entry['caption']}" for entry in caption_entries
            )
            updated_content = f"{updated_content}\n\nImage captions:\n{appendix}".strip()

        metadata["image_captions"] = caption_entries
        metadata["caption_model"] = (
            str(self.model or getattr(self.vision_llm, "model", None) or "unknown")
            if caption_entries or failures
            else metadata.get("caption_model")
        )
        metadata["caption_prompt_version"] = self.prompt_version
        if failures:
            metadata["has_unprocessed_images"] = True
            metadata["image_caption_failures"] = failures
        else:
            metadata.pop("has_unprocessed_images", None)
            metadata.pop("image_caption_failures", None)

        item["metadata"] = metadata
        item["image_refs"] = image_refs
        item["content"] = updated_content
        return item

    def _caption_image(self, chunk: dict[str, Any], image: dict[str, Any]) -> dict[str, Any]:
        if self.vision_llm is None:
            raise ValueError("vision llm is unavailable")

        image_id = str(image.get("id") or "").strip()
        image_path = str(image.get("path") or "").strip()
        if not image_path:
            raise ValueError(f"image path is missing for {image_id or 'unknown image'}")

        result = generate_image_caption(
            image_path,
            vision_llm=self.vision_llm,
            cache_repository=self.cache_repository,
            model=self.model,
            prompt=self.prompt,
            prompt_version=self.prompt_version,
            context=str(chunk.get("content", "")),
        )
        return {
            "image_id": image_id,
            "caption": result["caption"],
            "image_hash": result["image_hash"],
            "model": result["model"],
            "prompt_version": result["prompt_version"],
            "cached": result["cached"],
            "path": image_path,
        }

    @staticmethod
    def _collect_relevant_image_ids(
        image_refs: list[str],
        occurrences: list[dict[str, Any]],
    ) -> list[str]:
        resolved: list[str] = []
        for image_id in image_refs + [str(item.get("image_id") or "") for item in occurrences]:
            if image_id and image_id not in resolved:
                resolved.append(image_id)
        return resolved

    @staticmethod
    def _inject_occurrence_captions(
        content: str,
        *,
        occurrences: list[dict[str, Any]],
        captions_by_id: dict[str, str],
    ) -> str:
        updated = content
        for occurrence in sorted(
            occurrences,
            key=lambda item: int(item.get("text_offset", 0)),
            reverse=True,
        ):
            image_id = str(occurrence.get("image_id") or "").strip()
            caption = captions_by_id.get(image_id)
            if not caption:
                continue
            start = int(occurrence.get("text_offset", 0))
            length = int(occurrence.get("text_length", 0))
            placeholder = updated[start : start + length] or f"[IMAGE: {image_id}]"
            injected = f"{placeholder}\n[Image caption: {caption}]"
            updated = f"{updated[:start]}{injected}{updated[start + length:]}"
        return updated
