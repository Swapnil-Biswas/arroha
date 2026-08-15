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

from app.config import (
    BM25_WEIGHT,
    DENSE_WEIGHT,
    MIN_RETRIEVAL_SCORE,
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
    Hybrid retriever combining dense semantic matching and sparse lexical matching.
    """

    def __init__(
        self,
        vector_retriever: Optional[VectorRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        dense_weight: float = DENSE_WEIGHT,
        bm25_weight: float = BM25_WEIGHT,
        min_score: float = MIN_RETRIEVAL_SCORE,
    ) -> None:
        self.vector_retriever = vector_retriever or VectorRetriever()
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.min_score = min_score

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
    ) -> tuple[list[SourceDocument], dict[str, float]]:
        """
        Execute hybrid retrieval.
        Returns (list_of_SourceDocuments, latency_dict).
        """
        w_dense = dense_weight if dense_weight is not None else self.dense_weight
        w_bm25 = bm25_weight if bm25_weight is not None else self.bm25_weight

        # Normalize weights to sum to 1.0
        total_w = w_dense + w_bm25
        if total_w > 0:
            w_dense /= total_w
            w_bm25 /= total_w

        # Candidate pool multiplier
        candidate_k = max(top_k * 3, 15)

        # 1. Execute Dense Vector Retrieval
        vec_results, embed_ms, vec_search_ms = self.vector_retriever.search(query, top_k=candidate_k)

        # 2. Execute BM25 Retrieval
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

            if fusion_method == "rrf":
                # Reciprocal Rank Fusion (k=60)
                rrf_k = 60
                d_rank = entry["dense_rank"]
                b_rank = entry["bm25_rank"]
                fused = (1.0 / (rrf_k + d_rank)) + (1.0 / (rrf_k + b_rank))
            else:
                # Weighted linear score fusion gated by absolute dense similarity
                relative_fused = (w_dense * norm_dense[idx]) + (w_bm25 * norm_bm25[idx])
                # Scale by absolute cosine score so out-of-domain queries stay low
                fused = relative_fused * max(raw_d, 0.0)

            # Filter out candidates with very low absolute relevance
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

        latencies = {
            "query_embed_ms": round(embed_ms, 2),
            "vector_retrieval_ms": round(vec_search_ms, 2),
            "bm25_retrieval_ms": round(bm25_search_ms, 2),
            "hybrid_fusion_ms": round(fusion_ms, 2),
        }

        return sources, latencies
