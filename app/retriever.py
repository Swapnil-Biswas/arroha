"""
app/retriever.py
----------------
Module providing top-level search and warmup interface for retrieval benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from app.retrieval.vector import VectorRetriever

_retriever: Optional[VectorRetriever] = None


@dataclass
class SearchResponse:
    results: list[Any]
    embed_ms: float
    search_ms: float
    total_ms: float


def get_retriever() -> VectorRetriever:
    global _retriever
    if _retriever is None:
        _retriever = VectorRetriever()
    return _retriever


def warmup() -> None:
    """Warm up the embedding model and FAISS vector index."""
    r = get_retriever()
    r.search("warmup query", top_k=1)


def search(query: str, top_k: int = 5) -> SearchResponse:
    """Execute vector retrieval search and return structured timing response."""
    r = get_retriever()
    results, embed_ms, search_ms = r.search(query, top_k=top_k)
    return SearchResponse(
        results=results,
        embed_ms=embed_ms,
        search_ms=search_ms,
        total_ms=round(embed_ms + search_ms, 3),
    )
