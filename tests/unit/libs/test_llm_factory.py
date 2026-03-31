from __future__ import annotations

import pytest

from ragms.libs.factories.llm_factory import LLMFactory
from ragms.libs.providers.llm.openai_llm import OpenAILLM
from ragms.libs.providers.llm.qwen_llm import QwenLLM


def test_llm_factory_creates_configured_provider() -> None:
    llm = LLMFactory.create({"provider": "openai", "model": "gpt-4.1-mini", "temperature": 0.1, "api_key": "test-key"})
    assert isinstance(llm, OpenAILLM)

    llm = LLMFactory.create({"provider": "qwen", "model": "qwen-max"})
    assert isinstance(llm, QwenLLM)


def test_llm_factory_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="requires provider and model"):
        LLMFactory.create({"provider": "openai"})

    with pytest.raises(ValueError, match="Unknown llm provider"):
        LLMFactory.create({"provider": "missing", "model": "demo"})
