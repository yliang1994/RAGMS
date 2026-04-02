from __future__ import annotations

from ragms.ingestion_pipeline.transform import SemanticMetadataInjector, inject_semantic_metadata
from ragms.ingestion_pipeline.transform.services import MetadataService


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


def test_metadata_service_handles_short_text_without_explicit_heading() -> None:
    service = MetadataService()

    semantic = service.enrich(_chunk("API quota limits apply to batch jobs.") )

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
