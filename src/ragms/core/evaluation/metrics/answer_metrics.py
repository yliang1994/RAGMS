"""Deterministic answer-shape and citation metrics used before external evaluators."""

from __future__ import annotations

import re


_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def compute_citation_coverage(answer: str, citations: list[dict[str, object]]) -> float:
    """Return the fraction of expected citations that appear in the answer body."""

    if not citations:
        return 0.0
    markers = {f"[{citation.get('index')}]" for citation in citations if citation.get("index") is not None}
    if not markers:
        return 0.0
    used = {marker for marker in markers if marker in answer}
    return float(len(used) / len(markers))


def compute_answer_structure_score(answer: str) -> float:
    """Return a simple structure score rewarding non-empty, sentence-like answers."""

    text = answer.strip()
    if not text:
        return 0.0
    score = 0.0
    if len(text.split()) >= 3:
        score += 0.4
    if any(punct in text for punct in ".。!！?？"):
        score += 0.3
    if _CITATION_PATTERN.search(text):
        score += 0.3
    return min(score, 1.0)
