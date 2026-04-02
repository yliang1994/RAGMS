"""Lightweight DeepSeek-compatible LLM provider placeholder."""

from __future__ import annotations

from ragms.libs.providers.llm.openai_llm import OpenAILLM


class DeepSeekLLM(OpenAILLM):
    """Return deterministic text responses for the configured DeepSeek model."""

    provider_name = "deepseek"
