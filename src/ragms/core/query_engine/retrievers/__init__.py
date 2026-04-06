"""Retrieval backends for dense and sparse query routes."""

from __future__ import annotations

from .dense_retriever import DenseRetriever, DenseRetrieverError

__all__ = ["DenseRetriever", "DenseRetrieverError"]
