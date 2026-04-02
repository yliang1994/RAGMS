"""OpenAI SDK-backed embedding provider."""

from __future__ import annotations

from typing import Any

from openai import APIError, AuthenticationError, BadRequestError, OpenAI

from ragms.libs.abstractions import BaseEmbedding
from ragms.runtime.exceptions import RagMSError


class EmbeddingProviderError(RagMSError):
    """Raised when an embedding provider cannot produce usable vectors."""


class OpenAIEmbedding(BaseEmbedding):
    """Generate embeddings through the OpenAI Python SDK."""

    provider_display_name = "OpenAI"

    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        dimension: int | None = None,
        base_url: str | None = None,
        client: OpenAI | None = None,
    ) -> None:
        self.model = model.strip()
        self.api_key = api_key
        self.dimension = dimension
        self.base_url = base_url
        self._client = client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts and validate vector shape consistency."""

        if not texts:
            return []
        response = self._request_embeddings(texts)
        vectors = [self._coerce_embedding(item) for item in getattr(response, "data", [])]
        if len(vectors) != len(texts):
            raise EmbeddingProviderError("OpenAI returned an unexpected embedding count")
        self._validate_dimensions(vectors)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text and validate the resulting vector shape."""

        if not text:
            raise EmbeddingProviderError("OpenAI embedding input must not be empty")
        response = self._request_embeddings(text)
        vectors = [self._coerce_embedding(item) for item in getattr(response, "data", [])]
        if len(vectors) != 1:
            raise EmbeddingProviderError("OpenAI returned an unexpected embedding count")
        self._validate_dimensions(vectors)
        return vectors[0]

    def _request_embeddings(self, input_value: str | list[str]) -> Any:
        """Issue the embeddings request through the OpenAI client."""

        self._validate_configuration()
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "input": input_value,
            "model": self.model,
        }
        if self.dimension is not None:
            kwargs["dimensions"] = self.dimension
        try:
            return client.embeddings.create(**kwargs)
        except AuthenticationError as exc:
            raise EmbeddingProviderError("OpenAI authentication failed") from exc
        except BadRequestError as exc:
            raise EmbeddingProviderError(f"OpenAI rejected embedding request: {self.model}") from exc
        except APIError as exc:
            raise EmbeddingProviderError("OpenAI embedding request failed") from exc
        except Exception as exc:  # pragma: no cover - defensive boundary
            raise EmbeddingProviderError("OpenAI embedding request failed") from exc

    def _validate_configuration(self) -> None:
        """Validate required provider configuration before request execution."""

        if not self.api_key and self._client is None:
            raise EmbeddingProviderError("OpenAI api_key is required")
        if not self.model:
            raise EmbeddingProviderError("OpenAI model must not be empty")

    def _get_client(self) -> OpenAI:
        """Return a lazily initialized OpenAI client."""

        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    @staticmethod
    def _coerce_embedding(item: Any) -> list[float]:
        """Normalize a single embedding payload into a float vector."""

        embedding = getattr(item, "embedding", None)
        if embedding is None:
            raise EmbeddingProviderError("OpenAI returned an embedding item without vector data")
        return [float(value) for value in embedding]

    def _validate_dimensions(self, vectors: list[list[float]]) -> None:
        """Validate vector dimensions against request expectations and consistency."""

        if not vectors:
            raise EmbeddingProviderError("OpenAI returned no embeddings")
        lengths = {len(vector) for vector in vectors}
        if len(lengths) != 1:
            raise EmbeddingProviderError("OpenAI returned embeddings with inconsistent dimensions")
        actual_dimension = next(iter(lengths))
        if self.dimension is not None and actual_dimension != self.dimension:
            raise EmbeddingProviderError(
                f"OpenAI returned embedding dimension {actual_dimension}, expected {self.dimension}"
            )
