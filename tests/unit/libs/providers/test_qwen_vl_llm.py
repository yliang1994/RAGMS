from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import APIError

from ragms.libs.providers.vision_llms.gpt4o_vision_llm import VisionProviderError
from ragms.libs.providers.vision_llms.qwen_vl_llm import QwenVLLLM


class FakeChatCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.responses: list[object] = []
        self.error: Exception | None = None

    def create(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        if self.error:
            raise self.error
        if self.responses:
            return self.responses.pop(0)
        raise AssertionError("No fake response queued")


class FakeOpenAIClient:
    def __init__(self, completions: FakeChatCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


def _build_response(text: str) -> object:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def _build_api_error(message: str) -> APIError:
    request = httpx.Request("POST", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
    return APIError(message, request=request, body=None)


def _write_png(path: Path) -> Path:
    path.write_bytes(
        bytes.fromhex(
            "89504E470D0A1A0A"
            "0000000D49484452000000010000000108060000001F15C489"
            "0000000A49444154789C6360000002000154A24F5D00000000"
            "49454E44AE426082"
        )
    )
    return path


def test_qwen_vl_llm_captions_single_image_with_context(tmp_path: Path) -> None:
    image_path = _write_png(tmp_path / "chart.png")
    completions = FakeChatCompletions()
    completions.responses.append(_build_response("图表显示收入持续增长。"))
    llm = QwenVLLLM(
        api_key="test-key",
        client=FakeOpenAIClient(completions),
    )

    caption = llm.caption(image_path, prompt="请描述图表", context="季度收入")

    assert caption == "图表显示收入持续增长。"
    assert llm.model == "qwen-vl-max"
    assert llm.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    content = completions.calls[0]["messages"][0]["content"]
    assert content[0]["text"] == "请描述图表\nContext: 季度收入"


def test_qwen_vl_llm_captions_multiple_images(tmp_path: Path) -> None:
    first = _write_png(tmp_path / "first.png")
    second = _write_png(tmp_path / "second.png")
    completions = FakeChatCompletions()
    completions.responses.extend([
        _build_response("第一张图片说明"),
        _build_response("第二张图片说明"),
    ])
    llm = QwenVLLLM(
        api_key="test-key",
        client=FakeOpenAIClient(completions),
    )

    captions = llm.caption_batch([first, second], prompt="总结图片")

    assert captions == ["第一张图片说明", "第二张图片说明"]
    assert len(completions.calls) == 2


def test_qwen_vl_llm_rejects_low_quality_empty_response(tmp_path: Path) -> None:
    image_path = _write_png(tmp_path / "chart.png")
    completions = FakeChatCompletions()
    completions.responses.append(_build_response(""))
    llm = QwenVLLLM(
        api_key="test-key",
        client=FakeOpenAIClient(completions),
    )

    with pytest.raises(VisionProviderError, match="Qwen-VL returned an empty response"):
        llm.caption(image_path)


def test_qwen_vl_llm_maps_model_unavailable_failures(tmp_path: Path) -> None:
    image_path = _write_png(tmp_path / "chart.png")
    completions = FakeChatCompletions()
    completions.error = _build_api_error("model unavailable")
    llm = QwenVLLLM(
        api_key="test-key",
        client=FakeOpenAIClient(completions),
    )

    with pytest.raises(VisionProviderError, match="Qwen-VL request failed"):
        llm.caption(image_path)
