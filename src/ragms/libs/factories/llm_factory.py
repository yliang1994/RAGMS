from __future__ import annotations

from typing import Any

from ragms.libs.abstractions import BaseLLM
from ragms.libs.providers.llm.deepseek_llm import DeepSeekLLM
from ragms.libs.providers.llm.openai_llm import OpenAILLM
from ragms.libs.providers.llm.qwen_llm import QwenLLM


class LLMFactory:
    _REGISTRY = {
        "openai": OpenAILLM,
        "qwen": QwenLLM,
        "deepseek": DeepSeekLLM,
    }

    @classmethod
    def create(cls, config: dict[str, Any]) -> BaseLLM:
        provider = config.get("provider")
        model = config.get("model")
        if not provider or not model:
            raise ValueError("LLM config requires provider and model")
        try:
            llm_class = cls._REGISTRY[provider]
        except KeyError as exc:
            raise ValueError(f"Unknown llm provider: {provider}") from exc
        return llm_class(
            model=model,
            api_key=config.get("api_key"),
            temperature=config.get("temperature"),
        )

