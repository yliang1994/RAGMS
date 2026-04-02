"""OpenAI SDK-backed LLM provider."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from openai import APIError, AuthenticationError, BadRequestError, OpenAI, RateLimitError

from ragms.libs.abstractions import BaseLLM
from ragms.runtime.exceptions import RagMSError


class LLMProviderError(RagMSError):
    """Raised when an LLM provider cannot satisfy a generation request."""


class OpenAILLM(BaseLLM):
    """Generate text responses through the OpenAI Python SDK."""

    provider_name = "openai"
    provider_display_name = "OpenAI"

    def __init__(
        self,
        *,
        model: str = "gpt-4.1-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        client: OpenAI | None = None,
    ) -> None:
        self.model = model.strip()
        self.api_key = api_key
        self.base_url = base_url
        self._client = client

    def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:
        """Generate a complete text response via OpenAI chat completions."""

        self._validate_configuration()
        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=self._build_messages(prompt, system_prompt),
            )
        except AuthenticationError as exc:
            raise LLMProviderError(self._authentication_error_message()) from exc
        except BadRequestError as exc:
            raise LLMProviderError(self._bad_request_error_message()) from exc
        except RateLimitError as exc:
            raise LLMProviderError(self._rate_limit_error_message()) from exc
        except APIError as exc:
            raise LLMProviderError(self._request_error_message()) from exc
        except Exception as exc:  # pragma: no cover - defensive boundary
            raise LLMProviderError(self._request_error_message()) from exc

        content = self._extract_response_text(response)
        if not content:
            raise LLMProviderError(f"{self.provider_display_name} returned an empty response")
        return content

    def stream(self, prompt: str, *, system_prompt: str | None = None) -> Iterator[str]:
        """Stream a text response chunk by chunk via OpenAI chat completions."""

        self._validate_configuration()
        client = self._get_client()
        try:
            stream = client.chat.completions.create(
                model=self.model,
                messages=self._build_messages(prompt, system_prompt),
                stream=True,
            )
            yielded = False
            for chunk in stream:
                text = self._extract_stream_text(chunk)
                if text:
                    yielded = True
                    yield text
            if not yielded:
                raise LLMProviderError(f"{self.provider_display_name} returned an empty stream")
        except AuthenticationError as exc:
            raise LLMProviderError(self._authentication_error_message()) from exc
        except BadRequestError as exc:
            raise LLMProviderError(self._bad_request_error_message()) from exc
        except RateLimitError as exc:
            raise LLMProviderError(self._rate_limit_error_message()) from exc
        except APIError as exc:
            raise LLMProviderError(self._request_error_message()) from exc
        except LLMProviderError:
            raise
        except Exception as exc:  # pragma: no cover - defensive boundary
            raise LLMProviderError(self._request_error_message()) from exc

    def _validate_configuration(self) -> None:
        """Validate required configuration before issuing requests."""

        if not self.api_key and self._client is None:
            raise LLMProviderError(f"{self.provider_display_name} api_key is required")
        if not self.model:
            raise LLMProviderError(f"{self.provider_display_name} model must not be empty")

    def _get_client(self) -> OpenAI:
        """Return a lazily initialized OpenAI client."""

        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    @staticmethod
    def _build_messages(prompt: str, system_prompt: str | None) -> list[dict[str, str]]:
        """Build OpenAI chat-completion messages."""

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

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

    @staticmethod
    def _extract_stream_text(chunk: Any) -> str:
        """Extract incremental text from a streamed chat completion chunk."""

        choices = getattr(chunk, "choices", None) or []
        if not choices:
            return ""
        delta = getattr(choices[0], "delta", None)
        content = getattr(delta, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict)
            )
        return "" if content is None else str(content)

    def _authentication_error_message(self) -> str:
        """Return the provider-specific authentication failure message."""

        return f"{self.provider_display_name} authentication failed"

    def _bad_request_error_message(self) -> str:
        """Return the provider-specific bad-request failure message."""

        return f"{self.provider_display_name} rejected model or request: {self.model}"

    def _rate_limit_error_message(self) -> str:
        """Return the provider-specific rate-limit failure message."""

        return f"{self.provider_display_name} rate limit exceeded"

    def _request_error_message(self) -> str:
        """Return the provider-specific generic request failure message."""

        return f"{self.provider_display_name} request failed"
