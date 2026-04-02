"""Qwen OpenAI-compatible gateway provider."""

from __future__ import annotations

from ragms.libs.providers.llm.openai_llm import OpenAILLM


class QwenLLM(OpenAILLM):
    """Generate text via a Qwen endpoint that speaks the OpenAI protocol."""

    provider_name = "qwen"
    provider_display_name = "Qwen"

    def __init__(
        self,
        *,
        model: str = "qwen-plus",
        api_key: str | None = None,
        base_url: str | None = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        client: object | None = None,
    ) -> None:
        super().__init__(model=model, api_key=api_key, base_url=base_url, client=client)
