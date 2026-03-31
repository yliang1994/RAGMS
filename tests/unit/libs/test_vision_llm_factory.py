from __future__ import annotations

import pytest

from ragms.libs.factories.vision_llm_factory import VisionLLMFactory
from ragms.libs.providers.vision_llms.gpt4o_vision_llm import GPT4OVisionLLM
from ragms.libs.providers.vision_llms.qwen_vl_llm import QwenVLLM


def test_vision_llm_factory_creates_configured_provider() -> None:
    vision_llm = VisionLLMFactory.create({"provider": "openai", "model": "gpt-4.1-mini", "api_key": "test-key"})
    assert isinstance(vision_llm, GPT4OVisionLLM)


def test_vision_llm_factory_switches_by_language_or_deployment_env() -> None:
    zh_vision = VisionLLMFactory.create({"model": "qwen-vl-max", "api_key": "test-key"}, document_language="zh")
    assert isinstance(zh_vision, QwenVLLM)

    cn_vision = VisionLLMFactory.create({"model": "qwen-vl-max", "api_key": "test-key"}, deployment_env="cn")
    assert isinstance(cn_vision, QwenVLLM)

    en_vision = VisionLLMFactory.create({"model": "gpt-4.1-mini", "api_key": "test-key"}, document_language="en")
    assert isinstance(en_vision, GPT4OVisionLLM)


def test_vision_llm_factory_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="requires provider and model"):
        VisionLLMFactory.create({"provider": "openai"})

    with pytest.raises(ValueError, match="Unknown vision llm provider"):
        VisionLLMFactory.create({"provider": "missing", "model": "demo"})
