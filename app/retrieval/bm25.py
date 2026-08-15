"""
app/retrieval/bm25.py
---------------------
Sparse BM25 retrieval service for lexical matching.
Measures retrieval latency with microsecond resolution.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from app.config import RETRIEVAL_TOP_K
from indexing.bm25_index import BM25IndexManager

logger = logging.getLogger(__name__)


class BM25Retriever:
    """
    Retrieves top-K keyword-matching chunks using BM25Okapi.
    """

    def __init__(self, index_manager: Optional[BM25IndexManager] = None) -> None:
        self.index_manager = index_manager or BM25IndexManager()

        if not self.index_manager.is_ready:
            self.index_manager.load()

    @property
    def is_ready(self) -> bool:
        return self.index_manager.is_ready

    def search(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> tuple[list[tuple[dict[str, Any], float]], float]:
        """
        Execute BM25 search for a query string.
        Returns (results, search_latency_ms) where results are (chunk_dict, raw_bm25_score).
        """
        if not self.is_ready:
            if not self.index_manager.load():
                logger.warning("BM25 index is not ready.")
                return [], 0.0

        return self.index_manager.search(query=query, top_k=top_k)
