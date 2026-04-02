"""Qwen-VL OpenAI-compatible vision gateway provider."""

from __future__ import annotations

from ragms.libs.providers.vision_llms.gpt4o_vision_llm import GPT4oVisionLLM


class QwenVLLLM(GPT4oVisionLLM):
    """Generate image captions through a Qwen-VL compatible gateway."""

    provider_name = "qwen_vl"
    provider_display_name = "Qwen-VL"

    def __init__(
        self,
        *,
        model: str = "qwen-vl-max",
        api_key: str | None = None,
        base_url: str | None = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        client: object | None = None,
    ) -> None:
        super().__init__(model=model, api_key=api_key, base_url=base_url, client=client)
