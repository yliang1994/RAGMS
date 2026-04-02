from __future__ import annotations

import json

from ragms.ingestion_pipeline.transform import SemanticMetadataInjector, inject_semantic_metadata
from ragms.ingestion_pipeline.transform.services import MetadataService
from tests.fakes.fake_llm import FakeLLM


def _chunk(content: str, *, metadata: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "chunk_id": "chunk-1",
        "document_id": "doc-1",
        "content": content,
        "source_path": "docs/ops.md",
        "chunk_index": 0,
        "start_offset": 0,
        "end_offset": len(content),
        "image_refs": [],
        "metadata": dict(metadata or {}),
    }


def test_inject_semantic_metadata_adds_title_summary_and_tags() -> None:
    chunk = _chunk(
        "Access control approvals require manager review before production access is granted. "
        "Audit logging remains mandatory for every privileged action.",
        metadata={"section_path": ["Security", "Access Control"]},
    )

    result = inject_semantic_metadata([chunk])

    semantic = result[0]["metadata"]["semantic"]
    assert semantic["title"] == "Access Control"
    assert semantic["summary"].startswith("Access control approvals require manager review")
    assert "access" in semantic["tags"]
    assert "control" in semantic["tags"]
    assert result[0]["metadata"]["chunk_title"] == "Access Control"
    assert result[0]["metadata"]["chunk_tags"] == semantic["tags"]
    assert result[0]["metadata"]["metadata_enriched_by"] == "rule"


def test_metadata_service_handles_short_text_without_explicit_heading() -> None:
    service = MetadataService()

    semantic = service.enrich(_chunk("API quota limits apply to batch jobs."))

    assert semantic["title"] == "API quota limits apply to batch jobs."
    assert semantic["summary"] == "API quota limits apply to batch jobs."
    assert semantic["short_text"] is True
    assert semantic["tags"][:3] == ["api", "quota", "limits"]


def test_injector_preserves_existing_semantic_metadata() -> None:
    injector = SemanticMetadataInjector()
    chunk = _chunk(
        "Incident response workflow defines escalation and ownership.",
        metadata={
            "semantic": {
                "title": "Incident Response",
                "summary": "Existing summary",
                "tags": ["incident", "response"],
            }
        },
    )

    result = injector.transform([chunk])

    semantic = result[0]["metadata"]["semantic"]
    assert semantic["title"] == "Incident Response"
    assert semantic["summary"] == "Existing summary"
    assert semantic["tags"] == ["incident", "response"]


def test_metadata_service_uses_path_context_for_weakly_structured_list_chunks() -> None:
    service = MetadataService()

    semantic = service.enrich(
        _chunk(
            "- verify replica status\n- promote standby if lag exceeds threshold",
            metadata={"section_path": []},
        )
    )

    assert semantic["title"] == "ops"
    assert semantic["summary"] == "verify replica status; promote standby if lag exceeds threshold"
    assert semantic["tags"]
    assert "ops" in semantic["tags"]


def test_metadata_service_uses_nearby_context_when_heading_is_missing() -> None:
    service = MetadataService()

    semantic = service.enrich(
        _chunk(
            "This paragraph explains rollback sequencing and communication points.",
            metadata={"previous_heading": "Incident Rollback"},
        ),
        context={"parent_title": "Production Changes"},
    )

    assert semantic["title"] == "Incident Rollback"
    assert semantic["summary"].startswith("This paragraph explains rollback sequencing")
    assert "incident" in semantic["tags"]


def test_injector_keeps_chunk_shape_stable() -> None:
    injector = SemanticMetadataInjector()
    chunk = _chunk("On-call runbooks describe paging and rollback steps.")

    result = injector.transform([chunk])

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
    assert result[0]["content"] == "On-call runbooks describe paging and rollback steps."


def test_injector_applies_llm_metadata_enrichment_on_top_of_rule_result() -> None:
    service = MetadataService(
        enable_llm_enrich=True,
        llm=FakeLLM(
            [
                json.dumps(
                    {
                        "title": "Privileged Access Approval Workflow",
                        "summary": "Manager review and audit logging govern privileged access approval.",
                        "tags": ["approval", "access", "audit"],
                    },
                    ensure_ascii=False,
                )
            ]
        ),
        llm_model="fake-llm",
        llm_prompt_version="semantic_metadata_v2",
    )

    result = inject_semantic_metadata(
        [
            _chunk(
                "Access control approvals require manager review before production access is granted. "
                "Audit logging remains mandatory for every privileged action.",
                metadata={"section_path": ["Security", "Access Control"]},
            )
        ],
        service=service,
        enable_llm_enrich=True,
        context={"collection": "demo"},
    )

    semantic = result[0]["metadata"]["semantic"]
    assert semantic["title"] == "Privileged Access Approval Workflow"
    assert semantic["summary"] == (
        "Manager review and audit logging govern privileged access approval."
    )
    assert semantic["tags"] == ["approval", "access", "audit"]
    assert result[0]["metadata"]["metadata_enriched_by"] == "llm"
    assert result[0]["metadata"]["prompt_version"] == "semantic_metadata_v2"
    assert result[0]["metadata"]["model"] == "fake-llm"
    assert "fallback_reason" not in result[0]["metadata"]
    assert "Access Control" in str(service._llm.calls[0]["prompt"])


def test_injector_falls_back_to_rule_metadata_when_llm_response_is_invalid() -> None:
    service = MetadataService(
        enable_llm_enrich=True,
        llm=FakeLLM(['{"title":"Only title"}']),
        llm_prompt_version="semantic_metadata_v1",
    )

    result = inject_semantic_metadata(
        [_chunk("API quota limits apply to batch jobs.")],
        service=service,
        enable_llm_enrich=True,
    )

    semantic = result[0]["metadata"]["semantic"]
    assert semantic["title"] == "API quota limits apply to batch jobs."
    assert semantic["summary"] == "API quota limits apply to batch jobs."
    assert result[0]["metadata"]["metadata_enriched_by"] == "rule"
    assert "fallback_reason" in result[0]["metadata"]
    assert result[0]["metadata"]["prompt_version"] == "semantic_metadata_v1"


def test_injector_can_explicitly_disable_llm_enrichment() -> None:
    service = MetadataService(
        enable_llm_enrich=True,
        llm=FakeLLM(
            [
                json.dumps(
                    {
                        "title": "Should not be used",
                        "summary": "Should not be used",
                        "tags": ["unused"],
                    }
                )
            ]
        ),
    )

    result = inject_semantic_metadata(
        [_chunk("API quota limits apply to batch jobs.")],
        service=service,
        enable_llm_enrich=False,
    )

    assert result[0]["metadata"]["semantic"]["title"] == "API quota limits apply to batch jobs."
    assert result[0]["metadata"]["metadata_enriched_by"] == "rule"
    assert service._llm.calls == []
