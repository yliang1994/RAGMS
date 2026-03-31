from __future__ import annotations

import pytest

from ragms.libs.providers.vision_llms.gpt4o_vision_llm import GPT4OVisionLLM


def test_gpt4o_vision_llm_captions_single_image_with_context() -> None:
    llm = GPT4OVisionLLM(model="gpt-4o", api_key="test-key")

    result = llm.caption("image.png", prompt="describe chart", context="financial report")

    assert result["provider"] == "openai"
    assert result["caption"] == "gpt4o-vision-caption | describe chart | context=financial report"


def test_gpt4o_vision_llm_captions_batch_images() -> None:
    llm = GPT4OVisionLLM(model="gpt-4o", api_key="test-key")

    results = llm.caption_batch(["a.png", "b.png"], prompt="describe")

    assert len(results) == 2
    assert results[0]["image_ref"] == "a.png"
    assert results[1]["image_ref"] == "b.png"


def test_gpt4o_vision_llm_handles_invalid_config_and_image_errors() -> None:
    with pytest.raises(ValueError, match="API key is required"):
        GPT4OVisionLLM(model="gpt-4o", api_key=None)

    with pytest.raises(ValueError, match="Unsupported GPT-4o Vision model"):
        GPT4OVisionLLM(model="bad-model", api_key="test-key")

    llm = GPT4OVisionLLM(model="gpt-4o", api_key="test-key")
    with pytest.raises(ValueError, match="invalid image encoding"):
        llm.caption("data:image/png,broken")

    with pytest.raises(RuntimeError, match="upstream response error"):
        llm.caption("image.png", simulate_upstream_failure=True)
