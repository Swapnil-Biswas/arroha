"""
app/retrieval/reranker.py
-------------------------
Optional lightweight candidate reranker.
Can be toggled via config to preserve <200ms latency budget.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from app.config import ENABLE_RERANKER
from app.schemas.response import SourceDocument

logger = logging.getLogger(__name__)


class Reranker:
    """
    Reranker for second-stage candidate re-scoring.
    Defaults to pass-through when disabled to ensure <200ms pipeline budget.
    """

    def __init__(self, enabled: bool = ENABLE_RERANKER) -> None:
        self.enabled = enabled

    def rerank(
        self,
        query: str,
        candidates: list[SourceDocument],
        top_k: Optional[int] = None,
    ) -> tuple[list[SourceDocument], float]:
        """
        Rerank retrieved candidates.
        Returns (reranked_candidates, latency_ms).
        """
        t0 = time.perf_counter_ns()

        if not self.enabled or not candidates:
            latency_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            return candidates[:top_k] if top_k else candidates, latency_ms

        # Lightweight lexical overlap & length penalty re-scoring
        query_words = set(query.lower().split())
        scored: list[tuple[SourceDocument, float]] = []

        for doc in candidates:
            doc_words = set(doc.text.lower().split())
            overlap = len(query_words.intersection(doc_words)) / max(len(query_words), 1)
            # Combine base score with overlap boost
            boosted_score = (0.7 * doc.score) + (0.3 * overlap)
            doc.score = round(boosted_score, 4)
            scored.append((doc, boosted_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        reranked = [item[0] for item in scored]
        if top_k:
            reranked = reranked[:top_k]

        latency_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        return reranked, latency_ms
