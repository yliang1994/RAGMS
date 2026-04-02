"""Lightweight Qwen-compatible LLM provider placeholder."""

from __future__ import annotations

from ragms.libs.providers.llm.openai_llm import OpenAILLM


class QwenLLM(OpenAILLM):
    """Return deterministic text responses for the configured Qwen model."""

    provider_name = "qwen"
