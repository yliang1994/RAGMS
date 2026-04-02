from __future__ import annotations

import json

from ragms.ingestion_pipeline.transform import SemanticMetadataInjector, TransformPipeline
from ragms.ingestion_pipeline.transform.services import MetadataService
from tests.fakes.fake_llm import FakeLLM


def test_transform_pipeline_runs_rule_then_llm_metadata_enrichment() -> None:
    service = MetadataService(
        enable_llm_enrich=True,
        llm=FakeLLM(
            [
                json.dumps(
                    {
                        "title": "Rollback Communication Plan",
                        "summary": "Explains rollback sequencing and stakeholder communication.",
                        "tags": ["rollback", "communication", "operations"],
                    },
                    ensure_ascii=False,
                )
            ]
        ),
        llm_model="fake-llm",
        llm_prompt_version="semantic_metadata_v1",
    )
    pipeline = TransformPipeline(
        metadata_injector=SemanticMetadataInjector(
            service=service,
            enable_llm_enrich=True,
        )
    )

    result = pipeline.run(
        [
            {
                "content": "This paragraph explains rollback sequencing and communication points.",
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "source_path": "docs/incident-playbook.md",
                "chunk_index": 0,
                "start_offset": 0,
                "end_offset": 68,
                "source_ref": "doc-1",
                "image_refs": [],
                "metadata": {
                    "document_id": "doc-1",
                    "image_refs": [],
                    "previous_heading": "Incident Rollback",
                },
            }
        ],
        context={"parent_title": "Production Changes"},
    )

    assert len(result) == 1
    assert result[0]["content"] == "This paragraph explains rollback sequencing and communication points."
    assert result[0]["metadata"]["chunk_title"] == "Rollback Communication Plan"
    assert result[0]["metadata"]["chunk_summary"] == (
        "Explains rollback sequencing and stakeholder communication."
    )
    assert result[0]["metadata"]["chunk_tags"] == ["rollback", "communication", "operations"]
    assert result[0]["metadata"]["metadata_enriched_by"] == "llm"
    assert result[0]["metadata"]["prompt_version"] == "semantic_metadata_v1"
    assert result[0]["metadata"]["model"] == "fake-llm"
    assert "Incident Rollback" in str(service._llm.calls[0]["prompt"])
