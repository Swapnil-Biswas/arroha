"""
app/retrieval/vector.py
-----------------------
Dense vector retrieval service backed by FAISS and MultilingualEmbedder.
Measures embedding latency and FAISS search latency independently.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from app.config import RETRIEVAL_TOP_K
from indexing.embeddings import MultilingualEmbedder
from indexing.faiss_index import FAISSIndexManager

logger = logging.getLogger(__name__)


class VectorRetriever:
    """
    Retrieves top-K most semantically similar chunks using dense embeddings and FAISS.
    """

    def __init__(
        self,
        embedder: Optional[MultilingualEmbedder] = None,
        index_manager: Optional[FAISSIndexManager] = None,
    ) -> None:
        self.embedder = embedder or MultilingualEmbedder.get_instance()
        self.index_manager = index_manager or FAISSIndexManager()

        # Attempt to load index if not loaded
        if not self.index_manager.is_ready:
            self.index_manager.load()

    @property
    def is_ready(self) -> bool:
        return self.index_manager.is_ready

    def search(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> tuple[list[tuple[dict[str, Any], float]], float, float]:
        """
        Execute dense vector search for a query string.
        Returns (results, embed_latency_ms, search_latency_ms) where results are (chunk_dict, cosine_sim).
        """
        if not self.is_ready:
            # Try reloading in case index was recently built
            if not self.index_manager.load():
                logger.warning("Vector index is not ready.")
                return [], 0.0, 0.0

        # 1. Embed query
        query_vec, embed_latency_ms = self.embedder.embed_query(query)

        # 2. Search FAISS
        results, search_latency_ms = self.index_manager.search(query_vec, top_k=top_k)

        return results, embed_latency_ms, search_latency_ms
