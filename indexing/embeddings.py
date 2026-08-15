"""
indexing/embeddings.py
----------------------
Multilingual embedding generation with sentence-transformers.
Optimized for high throughput and ultra-low latency (<10ms per query on GPU).
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from app.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL_ID,
    NORMALIZE_EMBEDDINGS,
)

logger = logging.getLogger(__name__)


class MultilingualEmbedder:
    """
    Multilingual embedding generator wrapping sentence-transformers.
    Supports GPU acceleration, L2 normalization, batch encoding, and latency tracking.
    """

    _instance: Optional[MultilingualEmbedder] = None

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_ID,
        device: Optional[str] = None,
        normalize: bool = NORMALIZE_EMBEDDINGS,
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> None:
        self.model_name = model_name
        self.normalize = normalize
        self.batch_size = batch_size

        # Determine optimal device
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() and EMBEDDING_DEVICE == "cuda" else "cpu"

        logger.info("Embedding backend: SentenceTransformers")
        logger.info("Embedding device: %s", self.device)
        if self.device == "cuda" and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info("GPU: %s", gpu_name)
            logger.info("CUDA capability: %s", torch.cuda.get_device_capability(0))

        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.dim = getattr(self.model, "get_embedding_dimension", getattr(self.model, "get_sentence_embedding_dimension", lambda: EMBEDDING_DIM))() or EMBEDDING_DIM
        logger.info("Embedder initialized. Embedding dimension: %d", self.dim)

    @classmethod
    def get_instance(cls) -> MultilingualEmbedder:
        """Singleton pattern for fast reuse across pipeline calls."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def embed_query(self, query: str) -> tuple[np.ndarray, float]:
        """
        Embed a single user query.
        Returns (embedding_vector_1d, latency_ms).
        """
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter_ns()

        with torch.inference_mode():
            embedding = self.model.encode(
                query,
                batch_size=1,
                show_progress_bar=False,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
            )

        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter_ns() - t0) / 1_000_000.0

        if embedding.ndim == 2:
            embedding = embedding[0]
        return embedding.astype(np.float32), latency_ms

    def embed_documents(
        self,
        texts: list[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed a collection of documents in batches.
        Returns a 2D float32 numpy array of shape (N, dim).
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        with torch.inference_mode():
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
            )

        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        return embeddings.astype(np.float32)
