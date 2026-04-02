"""GPT-4o vision provider backed by the OpenAI Python SDK."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

from openai import APIError, AuthenticationError, BadRequestError, OpenAI

from ragms.libs.abstractions import BaseVisionLLM
from ragms.runtime.exceptions import RagMSError


class VisionProviderError(RagMSError):
    """Raised when a vision provider cannot produce image captions."""


class GPT4oVisionLLM(BaseVisionLLM):
    """Generate image captions through an OpenAI-compatible vision model."""

    provider_name = "gpt4o"
    provider_display_name = "GPT-4o Vision"

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        client: OpenAI | None = None,
    ) -> None:
        self.model = model.strip()
        self.api_key = api_key
        self.base_url = base_url
        self._client = client

    def caption(
        self,
        image_path: str | Path,
        *,
        prompt: str | None = None,
        context: str | None = None,
    ) -> str:
        """Generate a caption for a single image."""

        self._validate_configuration()
        path = Path(image_path)
        data_url = self._encode_image_data_url(path)
        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=self._build_messages(data_url=data_url, prompt=prompt, context=context),
            )
        except AuthenticationError as exc:
            raise VisionProviderError(f"{self.provider_display_name} authentication failed") from exc
        except BadRequestError as exc:
            raise VisionProviderError(
                f"{self.provider_display_name} rejected model or request: {self.model}"
            ) from exc
        except APIError as exc:
            raise VisionProviderError(f"{self.provider_display_name} request failed") from exc
        except Exception as exc:  # pragma: no cover - defensive boundary
            raise VisionProviderError(f"{self.provider_display_name} request failed") from exc

        content = self._extract_response_text(response)
        if not content:
            raise VisionProviderError(f"{self.provider_display_name} returned an empty response")
        return content

    def caption_batch(
        self,
        image_paths: list[str | Path],
        *,
        prompt: str | None = None,
        context: str | None = None,
    ) -> list[str]:
        """Generate captions for multiple images."""

        return [
            self.caption(image_path, prompt=prompt, context=context)
            for image_path in image_paths
        ]

    def _validate_configuration(self) -> None:
        """Validate required provider configuration."""

        if not self.api_key and self._client is None:
            raise VisionProviderError(f"{self.provider_display_name} api_key is required")
        if not self.model:
            raise VisionProviderError(f"{self.provider_display_name} model must not be empty")

    def _get_client(self) -> OpenAI:
        """Return a lazily initialized OpenAI client."""

        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def _encode_image_data_url(self, path: Path) -> str:
        """Encode an image into a data URL accepted by OpenAI vision models."""

        if not path.is_file():
            raise VisionProviderError(f"Image file does not exist: {path}")
        mime_type, _ = mimetypes.guess_type(path.name)
        if mime_type not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
            raise VisionProviderError(f"Unsupported image type for captioning: {path.suffix or '<no extension>'}")
        try:
            payload = base64.b64encode(path.read_bytes()).decode("ascii")
        except OSError as exc:  # pragma: no cover - defensive boundary
            raise VisionProviderError(f"Failed to read image file: {path}") from exc
        except UnicodeError as exc:  # pragma: no cover - defensive boundary
            raise VisionProviderError(f"Failed to encode image file: {path}") from exc
        if not payload:
            raise VisionProviderError(f"Image file is empty: {path}")
        return f"data:{mime_type};base64,{payload}"

    @staticmethod
    def _build_messages(
        *,
        data_url: str,
        prompt: str | None,
        context: str | None,
    ) -> list[dict[str, Any]]:
        """Build OpenAI chat-completion messages for vision captioning."""

        instructions = prompt or "Describe the image in a concise but informative way."
        if context:
            instructions = f"{instructions}\nContext: {context}"
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instructions},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        """Extract plain text from a chat completion response."""

        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict)
            )
        return "" if content is None else str(content)
