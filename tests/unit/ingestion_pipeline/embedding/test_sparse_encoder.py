from __future__ import annotations

from ragms.ingestion_pipeline.embedding import SparseEncoder, SparseEncodingError


def test_sparse_encoder_generates_stable_term_statistics() -> None:
    encoder = SparseEncoder(enable_jieba=False)

    encoded = encoder.encode(
        [{"content": "API quota limits and API budget alerts"}]
    )[0]

    assert encoded["tokens"] == ["api", "quota", "limits", "api", "budget", "alerts"]
    assert encoded["term_frequencies"] == {
        "alerts": 1,
        "api": 2,
        "budget": 1,
        "limits": 1,
        "quota": 1,
    }
    assert encoded["term_weights"]["api"] == 0.333333
    assert encoded["document_length"] == 6
    assert encoded["unique_terms"] == 5


def test_sparse_encoder_supports_chunk_like_objects_and_cache_reuse() -> None:
    encoder = SparseEncoder(enable_jieba=False, batch_size=1)

    first = encoder.encode([{"content": "Batch retry policy"}])[0]
    second = encoder.encode([{"content": "Batch retry policy"}])[0]

    assert first == second
    assert len(encoder._cache) == 1


def test_sparse_encoder_returns_explicit_empty_representation_for_blank_text() -> None:
    encoder = SparseEncoder(enable_jieba=False)

    encoded = encoder.encode(["   "])[0]

    assert encoded["tokens"] == []
    assert encoded["term_frequencies"] == {}
    assert encoded["term_weights"] == {}
    assert encoded["document_length"] == 0
    assert encoded["unique_terms"] == 0


def test_sparse_encoder_can_disable_case_normalization() -> None:
    encoder = SparseEncoder(enable_jieba=False, normalize_case=False)

    encoded = encoder.encode(["API Api api"])[0]

    assert encoded["tokens"] == ["API", "Api", "api"]
    assert encoded["term_frequencies"] == {"API": 1, "Api": 1, "api": 1}


def test_sparse_encoder_extracts_cjk_tokens() -> None:
    encoder = SparseEncoder(enable_jieba=True)

    encoded = encoder.encode(["混合检索和批处理优化"])[0]

    assert encoded["document_length"] >= 2
    assert any(token for token in encoded["tokens"] if any("\u4e00" <= char <= "\u9fff" for char in token))


def test_sparse_encoder_rejects_unknown_input_shape() -> None:
    encoder = SparseEncoder()

    encoded_error = None
    try:
        encoder.encode([object()])
    except SparseEncodingError as exc:
        encoded_error = exc

    assert encoded_error is not None
    assert "Sparse encoder expected strings or chunk-like objects" in str(encoded_error)
