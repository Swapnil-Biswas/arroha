"""
app/retrieval/hybrid.py
-----------------------
Hybrid retrieval fusing dense vector search (FAISS) and sparse keyword search (BM25).
Supports min-max score normalization + weighted linear combination as well as Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from app.config import (
    BM25_WEIGHT,
    DENSE_WEIGHT,
    ENABLE_RAG_CACHE,
    MIN_RETRIEVAL_SCORE,
    PARALLEL_HYBRID_SEARCH,
    RETRIEVAL_TOP_K,
)
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.vector import VectorRetriever
from app.schemas.response import SourceDocument

logger = logging.getLogger(__name__)


def min_max_normalize(scores: list[float]) -> list[float]:
    """Normalize a list of positive scores into [0.0, 1.0]."""
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    if max_s == min_s:
        return [1.0 if max_s > 0 else 0.0 for _ in scores]
    return [(s - min_s) / (max_s - min_s) for s in scores]


class HybridRetriever:
    """
    Hybrid retriever combining dense semantic matching and sparse lexical matching
    with high-speed LRU result caching.
    """

    def __init__(
        self,
        vector_retriever: Optional[VectorRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        dense_weight: float = DENSE_WEIGHT,
        bm25_weight: float = BM25_WEIGHT,
        min_score: float = MIN_RETRIEVAL_SCORE,
        cache_size: int = 4096,
    ) -> None:
        self.vector_retriever = vector_retriever or VectorRetriever()
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.min_score = min_score
        self._cache_size = cache_size
        self._cache: OrderedDict[tuple[str, int, float, float, str], list[SourceDocument]] = OrderedDict()
        self._cache_lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        return self.vector_retriever.is_ready or self.bm25_retriever.is_ready

    def search(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
        dense_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
        fusion_method: str = "weighted",  # 'weighted' or 'rrf'
        target_language: Optional[str] = None,
    ) -> tuple[list[SourceDocument], dict[str, float]]:
        """
        Execute hybrid retrieval.
        Returns (list_of_SourceDocuments, latency_dict).
        """
        t_start = time.perf_counter_ns()
        norm_q = query.strip()
        w_dense = dense_weight if dense_weight is not None else self.dense_weight
        w_bm25 = bm25_weight if bm25_weight is not None else self.bm25_weight

        # Check LRU cache if enabled
        cache_key = (norm_q, top_k, round(w_dense, 2), round(w_bm25, 2), fusion_method)
        if ENABLE_RAG_CACHE:
            with self._cache_lock:
                if cache_key in self._cache:
                    cached_sources = self._cache[cache_key]
                    self._cache.move_to_end(cache_key)
                    hit_ms = (time.perf_counter_ns() - t_start) / 1_000_000.0
                    return cached_sources, {
                        "query_embed_ms": 0.0,
                        "vector_retrieval_ms": 0.0,
                        "bm25_retrieval_ms": 0.0,
                        "hybrid_fusion_ms": round(hit_ms, 2),
                    }

        # Normalize weights to sum to 1.0
        total_w = w_dense + w_bm25
        if total_w > 0:
            w_dense /= total_w
            w_bm25 /= total_w

        # Candidate pool multiplier
        candidate_k = max(top_k * 3, 15)

        if PARALLEL_HYBRID_SEARCH and self.bm25_retriever.is_ready and w_bm25 > 0:
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_vec = executor.submit(self.vector_retriever.search, query, top_k=candidate_k)
                future_bm25 = executor.submit(self.bm25_retriever.search, query, top_k=candidate_k)
                vec_results, embed_ms, vec_search_ms = future_vec.result()
                bm25_results, bm25_search_ms = future_bm25.result()
        else:
            # Sequential fallback
            vec_results, embed_ms, vec_search_ms = self.vector_retriever.search(query, top_k=candidate_k)
            bm25_results, bm25_search_ms = self.bm25_retriever.search(query, top_k=candidate_k)

        t_fusion_start = time.perf_counter_ns()

        # 3. Fuse candidates
        candidate_map: dict[str, dict[str, Any]] = {}

        # Process Vector candidates
        for rank, (chunk_data, score) in enumerate(vec_results):
            cid = chunk_data.get("chunk_id", chunk_data.get("id", f"vec_{rank}"))
            candidate_map[cid] = {
                "chunk": chunk_data,
                "dense_score": float(score),
                "dense_rank": rank + 1,
                "bm25_score": 0.0,
                "bm25_rank": 9999,
            }

        # Process BM25 candidates
        for rank, (chunk_data, score) in enumerate(bm25_results):
            cid = chunk_data.get("chunk_id", chunk_data.get("id", f"bm25_{rank}"))
            if cid in candidate_map:
                candidate_map[cid]["bm25_score"] = float(score)
                candidate_map[cid]["bm25_rank"] = rank + 1
            else:
                candidate_map[cid] = {
                    "chunk": chunk_data,
                    "dense_score": 0.0,
                    "dense_rank": 9999,
                    "bm25_score": float(score),
                    "bm25_rank": rank + 1,
                }

        # Score normalization & fusion
        all_cids = list(candidate_map.keys())
        raw_dense_scores = [candidate_map[cid]["dense_score"] for cid in all_cids]
        raw_bm25_scores = [candidate_map[cid]["bm25_score"] for cid in all_cids]

        norm_dense = min_max_normalize(raw_dense_scores)
        norm_bm25 = min_max_normalize(raw_bm25_scores)

        fused_candidates: list[tuple[str, float, dict[str, Any]]] = []

        for idx, cid in enumerate(all_cids):
            entry = candidate_map[cid]
            raw_d = entry["dense_score"]
            chunk_lang = str(entry["chunk"].get("language", "")).lower()

            if fusion_method == "rrf":
                # Reciprocal Rank Fusion (k=60)
                rrf_k = 60
                d_rank = entry["dense_rank"]
                b_rank = entry["bm25_rank"]
                fused = (1.0 / (rrf_k + d_rank)) + (1.0 / (rrf_k + b_rank))
            else:
                # Weighted linear score fusion
                fused = (w_dense * norm_dense[idx]) + (w_bm25 * norm_bm25[idx])

            # Language match boost if chunk matches query language
            if target_language:
                t_clean = target_language.lower()
                if (t_clean in ("en", "latin") and chunk_lang == "en") or (t_clean == chunk_lang):
                    fused += 0.20

            # Filter out candidates with zero relevance
            if raw_d >= self.min_score or entry["bm25_score"] > 0:
                fused_candidates.append((cid, fused, entry))

        # Sort descending by fused score
        fused_candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = fused_candidates[:top_k]

        fusion_ms = (time.perf_counter_ns() - t_fusion_start) / 1_000_000.0

        # Construct SourceDocument outputs
        sources: list[SourceDocument] = []
        for cid, score, entry in top_candidates:
            c = entry["chunk"]
            sources.append(
                SourceDocument(
                    doc_id=c.get("doc_id", cid),
                    text=c.get("text", ""),
                    language=c.get("language", "hi"),
                    score=round(score, 4),
                    dense_score=round(entry["dense_score"], 4),
                    bm25_score=round(entry["bm25_score"], 4),
                    query_id=c.get("query_id"),
                    passage_id=c.get("passage_id"),
                    is_selected=c.get("is_selected"),
                )
            )

        if ENABLE_RAG_CACHE:
            with self._cache_lock:
                if len(self._cache) >= self._cache_size:
                    self._cache.popitem(last=False)
                self._cache[cache_key] = sources

        latencies = {
            "query_embed_ms": round(embed_ms, 2),
            "vector_retrieval_ms": round(vec_search_ms, 2),
            "bm25_retrieval_ms": round(bm25_search_ms, 2),
            "hybrid_fusion_ms": round(fusion_ms, 2),
        }

        return sources, latencies
