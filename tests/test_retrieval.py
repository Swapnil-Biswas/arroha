"""
tests/test_retrieval.py
-----------------------
Unit and integration tests for FAISS vector search, BM25 keyword search,
and hybrid score fusion.
"""

import pytest
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.hybrid import HybridRetriever, min_max_normalize
from app.retrieval.reranker import Reranker
from app.retrieval.vector import VectorRetriever
from indexing.bm25_index import tokenize_multilingual


def test_tokenize_multilingual():
    hindi = "भारत की राजधानी नई दिल्ली है।"
    tokens = tokenize_multilingual(hindi)
    assert "भारत" in tokens
    assert "राजधानी" in tokens

    bengali = "রবীন্দ্রনাথ ঠাকুর সাহিত্য"
    b_tokens = tokenize_multilingual(bengali)
    assert "রবীন্দ্রনাথ" in b_tokens


def test_min_max_normalize():
    scores = [10.0, 20.0, 30.0]
    norm = min_max_normalize(scores)
    assert norm == [0.0, 0.5, 1.0]

    empty = min_max_normalize([])
    assert empty == []

    same = min_max_normalize([5.0, 5.0])
    assert same == [1.0, 1.0]


def test_retrieval_pipeline_execution():
    vector_ret = VectorRetriever()
    bm25_ret = BM25Retriever()
    hybrid_ret = HybridRetriever(vector_retriever=vector_ret, bm25_retriever=bm25_ret)

    if not hybrid_ret.is_ready:
        pytest.skip("Indexes not loaded on disk.")

    query = "भारत की राजधानी क्या है?"
    sources, latencies = hybrid_ret.search(query, top_k=3)

    assert len(sources) > 0
    assert "query_embed_ms" in latencies
    assert "vector_retrieval_ms" in latencies
    assert "bm25_retrieval_ms" in latencies
    assert "hybrid_fusion_ms" in latencies

    # Top candidate should be relevant
    top_doc = sources[0]
    assert top_doc.score > 0.0
    assert len(top_doc.text) > 0


def test_reranker_toggle():
    reranker_disabled = Reranker(enabled=False)
    hybrid_ret = HybridRetriever()

    if not hybrid_ret.is_ready:
        pytest.skip("Indexes not loaded on disk.")

    sources, _ = hybrid_ret.search("New Delhi history", top_k=2)
    reranked, lat_ms = reranker_disabled.rerank("New Delhi history", sources)
    assert len(reranked) == len(sources)
    assert lat_ms < 5.0
