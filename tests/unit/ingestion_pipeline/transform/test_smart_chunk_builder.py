from __future__ import annotations

import json

from ragms.ingestion_pipeline.transform.smart_chunk_builder import SmartChunkBuilder
from tests.fakes.fake_llm import FakeLLM


def _chunk(
    index: int,
    content: str,
    *,
    chunk_id: str | None = None,
    start_offset: int | None = None,
    end_offset: int | None = None,
    image_refs: list[str] | None = None,
) -> dict[str, object]:
    start = index * 100 if start_offset is None else start_offset
    end = start + len(content) if end_offset is None else end_offset
    return {
        "chunk_id": chunk_id or f"chunk-{index}",
        "document_id": "doc-1",
        "content": content,
        "source_path": "docs/handbook.md",
        "chunk_index": index,
        "start_offset": start,
        "end_offset": end,
        "image_refs": list(image_refs or []),
        "metadata": {
            "source_sha256": "abc123",
            "page": 1,
        },
    }


class FlakyLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, str | None]] = []
        self._attempt = 0

    def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt})
        self._attempt += 1
        if self._attempt == 1:
            raise TimeoutError("slow llm")
        return json.dumps(
            {
                "content": "Recovered refined chunk.",
                "rationale": "retry succeeded",
            },
            ensure_ascii=False,
        )


def test_denoise_removes_noise_lines_and_drops_empty_chunks() -> None:
    builder = SmartChunkBuilder()

    result = builder.denoise(
        [
            _chunk(0, "Page 1 of 3\n%%%%%%\nThis   is   the useful paragraph.\n1 / 3"),
            _chunk(1, "Page 2 of 3\n------\n%%%%%"),
        ]
    )

    assert len(result) == 1
    assert result[0]["content"] == "This is the useful paragraph."
    assert result[0]["chunk_id"] == "chunk-0"
    assert result[0]["metadata"]["refined_by"] == "rule"
    assert result[0]["metadata"]["smart_chunk"]["denoised"] is True
    assert result[0]["metadata"]["smart_chunk"]["removed_noise_lines"] == [
        "Page 1 of 3",
        "%%%%%%",
        "1 / 3",
    ]
    assert "normalized_whitespace" in result[0]["metadata"]["smart_chunk"]["rewrite_actions"]
    assert result[0]["metadata"]["smart_chunk"]["original_content_sha256"]


def test_rewrite_merges_neighboring_chunks_with_continuation_context() -> None:
    builder = SmartChunkBuilder(merge_below_chars=120)

    result = builder.rewrite(
        [
            _chunk(0, "This policy applies to all employees because", image_refs=["img-1"]),
            _chunk(1, "it defines the approval flow and exceptions.", image_refs=["img-2"]),
        ]
    )

    assert len(result) == 1
    assert result[0]["content"] == (
        "This policy applies to all employees because\n\n"
        "it defines the approval flow and exceptions."
    )
    assert result[0]["chunk_id"] == "chunk-0"
    assert result[0]["start_offset"] == 0
    assert result[0]["end_offset"] == 144
    assert result[0]["image_refs"] == ["img-1", "img-2"]
    assert result[0]["metadata"]["refined_by"] == "rule"
    assert result[0]["metadata"]["smart_chunk"]["merged"] is True
    assert result[0]["metadata"]["smart_chunk"]["merged_from"] == ["chunk-0", "chunk-1"]
    assert "merged_neighbor_context" in result[0]["metadata"]["smart_chunk"]["rewrite_actions"]


def test_rewrite_keeps_boundaries_for_standalone_chunks() -> None:
    builder = SmartChunkBuilder(merge_below_chars=40)

    result = builder.rewrite(
        [
            _chunk(0, "This section is complete."),
            _chunk(
                1,
                "Eligibility requirements start here and continue with enough detail to stay independent.",
            ),
        ]
    )

    assert len(result) == 2
    assert result[0]["content"] == "This section is complete."
    assert result[1]["content"].startswith("Eligibility requirements")
    assert result[0]["metadata"]["smart_chunk"]["merged"] is False
    assert "refined_by" not in result[0]["metadata"]


def test_denoise_preserves_markdown_lists_tables_and_code_blocks() -> None:
    builder = SmartChunkBuilder()

    result = builder.denoise(
        [
            _chunk(
                0,
                "#\nPolicy Overview\n\n"
                "- keep first item\n"
                "- keep second item\n\n"
                "| Name | Value |\n"
                "| ---- | ----- |\n"
                "| Foo | Bar |\n\n"
                "```\n"
                "def hello():\n"
                "    return 42\n"
                "```\n",
            )
        ]
    )

    assert result[0]["content"] == (
        "# Policy Overview\n\n"
        "- keep first item\n"
        "- keep second item\n\n"
        "| Name | Value |\n"
        "| ---- | ----- |\n"
        "| Foo | Bar |\n\n"
        "```\n"
        "def hello():\n"
        "    return 42\n"
        "```"
    )
    assert "repaired_heading_break" in result[0]["metadata"]["smart_chunk"]["rewrite_actions"]


def test_transform_runs_denoise_then_rewrite_and_records_source_grouping() -> None:
    builder = SmartChunkBuilder(merge_below_chars=120)

    result = builder.transform(
        [
            _chunk(0, "%%%%\nThe on-\nboarding checklist includes"),
            _chunk(1, "the required approvals and system setup.", image_refs=["img-3"]),
        ]
    )

    assert len(result) == 1
    assert set(result[0]) == {
        "chunk_id",
        "document_id",
        "content",
        "source_path",
        "chunk_index",
        "start_offset",
        "end_offset",
        "image_refs",
        "metadata",
    }
    assert result[0]["content"] == (
        "The onboarding checklist includes\n\n"
        "the required approvals and system setup."
    )
    assert result[0]["chunk_id"] == "chunk-0"
    assert result[0]["metadata"]["image_refs"] == ["img-3"]
    assert result[0]["metadata"]["refined_by"] == "rule"
    assert result[0]["metadata"]["smart_chunk"]["source_chunk_count"] == 2
    assert "repaired_hyphen_break" in result[0]["metadata"]["smart_chunk"]["rewrite_actions"]


def test_transform_applies_llm_refinement_on_top_of_rule_result() -> None:
    llm = FakeLLM(
        [
            json.dumps(
                {
                    "content": "The onboarding checklist covers approvals and system setup.",
                    "rationale": "Merged adjacent context into one concise chunk.",
                },
                ensure_ascii=False,
            )
        ]
    )
    builder = SmartChunkBuilder(
        merge_below_chars=120,
        enable_llm_refine=True,
        llm=llm,
        llm_model="fake-llm",
        llm_prompt_version="smart_chunk_refine_v2",
    )

    result = builder.transform(
        [
            _chunk(0, "%%%%\nThe on-\nboarding checklist includes"),
            _chunk(1, "the required approvals and system setup."),
        ],
        context={"collection": "demo"},
    )

    assert len(result) == 1
    assert result[0]["content"] == "The onboarding checklist covers approvals and system setup."
    assert result[0]["metadata"]["refined_by"] == "llm"
    assert result[0]["metadata"]["smart_chunk"]["llm_refined"] is True
    assert result[0]["metadata"]["smart_chunk"]["prompt_version"] == "smart_chunk_refine_v2"
    assert result[0]["metadata"]["smart_chunk"]["model"] == "fake-llm"
    assert result[0]["metadata"]["smart_chunk"]["llm_rationale"] == (
        "Merged adjacent context into one concise chunk."
    )
    assert "%%%%" not in str(llm.calls[0]["prompt"])
    assert "The onboarding checklist includes" in str(llm.calls[0]["prompt"])


def test_refine_with_llm_falls_back_to_rule_result_when_llm_output_is_invalid() -> None:
    builder = SmartChunkBuilder(
        merge_below_chars=120,
        enable_llm_refine=True,
        llm=FakeLLM(["not json"]),
    )

    result = builder.transform(
        [
            _chunk(0, "This policy applies to all employees because"),
            _chunk(1, "it defines the approval flow and exceptions."),
        ]
    )

    assert len(result) == 1
    assert result[0]["content"] == (
        "This policy applies to all employees because\n\n"
        "it defines the approval flow and exceptions."
    )
    assert result[0]["metadata"]["refined_by"] == "rule"
    assert "fallback_reason" in result[0]["metadata"]
    assert result[0]["metadata"]["smart_chunk"]["llm_refined"] is False


def test_refine_with_llm_retries_and_uses_recovered_response() -> None:
    llm = FlakyLLM()
    builder = SmartChunkBuilder(
        enable_llm_refine=True,
        llm=llm,
        llm_max_retries=2,
    )

    result = builder.transform([_chunk(0, "Short chunk that should be refined")])

    assert result[0]["content"] == "Recovered refined chunk."
    assert result[0]["metadata"]["refined_by"] == "llm"
    assert len(llm.calls) == 2
