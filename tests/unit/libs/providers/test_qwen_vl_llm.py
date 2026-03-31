from __future__ import annotations

import pytest

from ragms.libs.providers.vision_llms.qwen_vl_llm import QwenVLLM


def test_qwen_vl_llm_captions_single_image_and_batch() -> None:
    llm = QwenVLLM(model="qwen-vl-max", api_key="test-key")

    single = llm.caption("chart.png", prompt="描述图表", context="中文财报")
    batch = llm.caption_batch(["a.png", "b.png"], prompt="描述")

    assert single["caption"] == "qwen-vl-caption | 描述图表 | context=中文财报"
    assert single["quality"] == "ok"
    assert len(batch) == 2
    assert batch[1]["image_ref"] == "b.png"


def test_qwen_vl_llm_handles_invalid_config_and_quality_failures() -> None:
    with pytest.raises(ValueError, match="API key is required"):
        QwenVLLM(model="qwen-vl-max", api_key=None)

    with pytest.raises(ValueError, match="Unsupported Qwen-VL model"):
        QwenVLLM(model="bad-model", api_key="test-key")

    llm = QwenVLLM(model="qwen-vl-max", api_key="test-key")
    low_quality = llm.caption("chart.png", simulate_low_quality=True)
    assert low_quality["quality"] == "low"
    assert low_quality["caption"] == "qwen-vl-caption-insufficient-detail"

    with pytest.raises(RuntimeError, match="model unavailable"):
        llm.caption("chart.png", simulate_model_unavailable=True)
