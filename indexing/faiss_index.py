"""
indexing/faiss_index.py
-----------------------
FAISS vector index manager supporting persistent storage,
normalized cosine similarity search, and low-latency in-memory queries.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np

from app.config import FAISS_INDEX_PATH, FAISS_METADATA_PATH
from ingestion.models import Chunk

logger = logging.getLogger(__name__)


class FAISSIndexManager:
    """
    Manages FAISS dense vector index and chunk metadata.
    """

    def __init__(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        metadata_path: Path = FAISS_METADATA_PATH,
        dim: int = 384,
    ) -> None:
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.dim = dim
        self.index: Optional[faiss.Index] = None
        self.metadata: list[dict[str, Any]] = []

    @property
    def is_ready(self) -> bool:
        return self.index is not None and self.index.ntotal > 0

    @property
    def count(self) -> int:
        return self.index.ntotal if self.index is not None else 0

    def build_index(
        self,
        embeddings: np.ndarray,
        chunks: list[Chunk],
        index_type: str = "FlatIP",
    ) -> None:
        """
        Build and populate a FAISS vector index from embeddings and chunks.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings")

        n, d = embeddings.shape
        self.dim = d
        logger.info("Building FAISS '%s' index with %d vectors of dimension %d...", index_type, n, d)

        if index_type == "FlatIP":
            # Exact inner product (cosine similarity on L2-normalized vectors)
            self.index = faiss.IndexFlatIP(d)
        elif index_type == "HNSW":
            # Fast Approximate Nearest Neighbor for large corpora
            self.index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = faiss.IndexFlatIP(d)

        # Add vectors
        self.index.add(embeddings.astype(np.float32))

        # Store metadata
        self.metadata = [chunk.model_dump() for chunk in chunks]
        logger.info("FAISS index built successfully. Total entries: %d", self.index.ntotal)

    def save(self) -> None:
        """Persist FAISS index and metadata to disk."""
        if self.index is None:
            raise RuntimeError("Cannot save uninitialized FAISS index.")

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            for item in self.metadata:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info("Saved FAISS index to %s and metadata to %s", self.index_path, self.metadata_path)

    def load(self) -> bool:
        """Load FAISS index and metadata from disk into memory."""
        if not self.index_path.exists() or not self.metadata_path.exists():
            logger.warning("FAISS index or metadata file not found at %s", self.index_path)
            return False

        logger.info("Loading FAISS index from %s...", self.index_path)
        self.index = faiss.read_index(str(self.index_path))

        self.metadata = []
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.metadata.append(json.loads(line))

        logger.info("Loaded FAISS index with %d vectors and %d metadata records.", self.index.ntotal, len(self.metadata))
        return True

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> tuple[list[tuple[dict[str, Any], float]], float]:
        """
        Perform vector similarity search.
        Returns (list_of_(chunk_dict, score), search_latency_ms).
        """
        if not self.is_ready:
            return [], 0.0

        t0 = time.perf_counter_ns()

        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)

        # FAISS search
        scores, indices = self.index.search(query_vector.astype(np.float32), k=min(top_k, self.index.ntotal))
        latency_ms = (time.perf_counter_ns() - t0) / 1_000_000.0

        results: list[tuple[dict[str, Any], float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0 and idx < len(self.metadata):
                results.append((self.metadata[idx], float(score)))

        return results, latency_ms
