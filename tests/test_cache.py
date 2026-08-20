"""
tests/test_cache.py
-------------------
Unit tests for RAGQueryCache.
"""

from app.cache import RAGQueryCache
from app.schemas.response import GroundingResult, LatencyBreakdown, RAGResponse


def test_rag_query_cache():
    cache = RAGQueryCache(capacity=2, ttl_seconds=10.0)

    resp = RAGResponse(
        query="what is RAG?",
        detected_language="en",
        answer="RAG is Retrieval-Augmented Generation.",
        is_refusal=False,
        grounding=GroundingResult(is_grounded=True, grounding_score=0.9),
        sources=[],
        latency=LatencyBreakdown(total_ms=100.0),
        request_id="test1",
    )

    # Put in cache
    cache.put("what is RAG?", "en", resp)

    # Fetch from cache
    cached = cache.get("what is RAG?", "en")
    assert cached is not None
    assert cached.answer == "RAG is Retrieval-Augmented Generation."
    assert cached.latency.total_ms < 5.0
    assert cached.debug_info.get("cache_hit") is True

    # Check cache miss for unknown query
    assert cache.get("unknown query", "en") is None

    # Check stats
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["size"] == 1


if __name__ == "__main__":
    test_rag_query_cache()
    print("ALL CACHE TESTS PASSED!")
