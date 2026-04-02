"""Lightweight Qwen-VL provider placeholder."""

from __future__ import annotations

from ragms.libs.providers.vision_llms.gpt4o_vision_llm import GPT4oVisionLLM


class QwenVLLLM(GPT4oVisionLLM):
    """Generate deterministic captions for the configured Qwen-VL model."""

    provider_name = "qwen_vl"
