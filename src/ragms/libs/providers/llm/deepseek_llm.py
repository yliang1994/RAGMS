"""DeepSeek OpenAI-compatible gateway provider."""

from __future__ import annotations

from ragms.libs.providers.llm.openai_llm import OpenAILLM


class DeepSeekLLM(OpenAILLM):
    """Generate text via a DeepSeek endpoint that speaks the OpenAI protocol."""

    provider_name = "deepseek"
    provider_display_name = "DeepSeek"

    def __init__(
        self,
        *,
        model: str = "deepseek-chat",
        api_key: str | None = None,
        base_url: str | None = "https://api.deepseek.com/v1",
        client: object | None = None,
    ) -> None:
        super().__init__(model=model, api_key=api_key, base_url=base_url, client=client)
