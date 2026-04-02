from __future__ import annotations

import json

from ragms.ingestion_pipeline.transform import TransformPipeline
from ragms.ingestion_pipeline.transform.smart_chunk_builder import SmartChunkBuilder
from tests.fakes.fake_llm import FakeLLM


def test_transform_pipeline_runs_rule_then_llm_smart_refinement() -> None:
    llm = FakeLLM(
        [
            json.dumps(
                {
                    "content": "The approval policy explains the workflow, scope, and exceptions.",
                    "rationale": "Expanded the chunk into a self-contained statement.",
                },
                ensure_ascii=False,
            )
        ]
    )
    pipeline = TransformPipeline(
        smart_chunk_builder=SmartChunkBuilder(
            merge_below_chars=120,
            enable_llm_refine=True,
            llm=llm,
            llm_model="fake-llm",
            llm_prompt_version="smart_chunk_refine_v1",
        )
    )

    result = pipeline.run(
        [
            {
                "content": "Page 1 of 3\nThis policy applies to all employees because",
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "source_path": "docs/policy.md",
                "chunk_index": 0,
                "start_offset": 0,
                "end_offset": 56,
                "source_ref": "doc-1",
                "image_refs": [],
                "metadata": {"document_id": "doc-1", "image_refs": []},
            },
            {
                "content": "it defines the approval flow and exceptions.",
                "chunk_id": "chunk-2",
                "document_id": "doc-1",
                "source_path": "docs/policy.md",
                "chunk_index": 1,
                "start_offset": 57,
                "end_offset": 102,
                "source_ref": "doc-1",
                "image_refs": [],
                "metadata": {"document_id": "doc-1", "image_refs": []},
            },
        ],
        context={"collection": "demo"},
    )

    assert len(result) == 1
    assert result[0]["chunk_id"] == "chunk-1"
    assert result[0]["content"] == (
        "The approval policy explains the workflow, scope, and exceptions."
    )
    assert result[0]["metadata"]["refined_by"] == "llm"
    assert result[0]["metadata"]["smart_chunk"]["merged_from"] == ["chunk-1", "chunk-2"]
    assert result[0]["metadata"]["smart_chunk"]["prompt_version"] == "smart_chunk_refine_v1"
    assert result[0]["metadata"]["smart_chunk"]["model"] == "fake-llm"
    assert "Page 1 of 3" not in str(llm.calls[0]["prompt"])
