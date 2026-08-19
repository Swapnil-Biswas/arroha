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
from transformers import AutoModel, AutoTokenizer

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
    Multilingual embedding generator with optimized direct PyTorch CUDA hot-path.
    Supports GPU acceleration (FP16), L2 normalization, batch encoding, and latency tracking.
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

        logger.info("Embedding backend: Direct PyTorch FP16 (CUDA hot-path) + SentenceTransformers")
        logger.info("Embedding device: %s", self.device)
        if self.device == "cuda" and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info("GPU: %s", gpu_name)
            logger.info("CUDA capability: %s", torch.cuda.get_device_capability(0))

        # 1. Initialize tokenizer and direct PyTorch transformer model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            if self.device == "cuda" and torch.cuda.is_available():
                self.torch_model = AutoModel.from_pretrained(self.model_name, local_files_only=True).to(self.device).half().eval()
            else:
                self.torch_model = AutoModel.from_pretrained(self.model_name, local_files_only=True).to(self.device).eval()
            self.model = SentenceTransformer(self.model_name, device=self.device, local_files_only=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.device == "cuda" and torch.cuda.is_available():
                self.torch_model = AutoModel.from_pretrained(self.model_name).to(self.device).half().eval()
            else:
                self.torch_model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
            self.model = SentenceTransformer(self.model_name, device=self.device)

        self.dim = getattr(self.model, "get_embedding_dimension", getattr(self.model, "get_sentence_embedding_dimension", lambda: EMBEDDING_DIM))() or EMBEDDING_DIM
        logger.info("Embedder initialized. Embedding dimension: %d", self.dim)

        # 3. Persistent GPU warm-up to initialize CUDA kernels and execution graph
        self._warmup()

    def _warmup(self) -> None:
        """Warm up CUDA execution stream and memory caches."""
        try:
            for _ in range(10):
                encoded = self.tokenizer(
                    "Warmup query for CUDA kernel initialization and GPU clock boosting",
                    padding=True,
                    truncation=True,
                    max_length=64,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(self.device, non_blocking=True)
                attn_mask = encoded["attention_mask"].to(self.device, non_blocking=True)
                with torch.inference_mode():
                    out = self.torch_model(input_ids=input_ids, attention_mask=attn_mask)
                    token_embeddings = out[0]
                    if self.device == "cuda":
                        mask_exp = attn_mask.unsqueeze(-1).expand(token_embeddings.size()).half()
                    else:
                        mask_exp = attn_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    sum_emb = torch.sum(token_embeddings * mask_exp, 1)
                    sum_m = torch.clamp(mask_exp.sum(1), min=1e-9)
                    pooled = sum_emb / sum_m
                    _ = torch.nn.functional.normalize(pooled, p=2, dim=1)
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as exc:
            logger.warning("Embedder warmup warning: %s", exc)

    @classmethod
    def get_instance(cls) -> MultilingualEmbedder:
        """Singleton pattern for fast reuse across pipeline calls."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def embed_query(self, query: str) -> tuple[np.ndarray, float]:
        """
        Embed a single user query using the direct PyTorch FP16 CUDA hot path.
        Returns (embedding_vector_1d, latency_ms).
        """
        t0 = time.perf_counter_ns()

        with torch.inference_mode():
            encoded = self.tokenizer(
                query,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device, non_blocking=True)
            attn_mask = encoded["attention_mask"].to(self.device, non_blocking=True)

            out = self.torch_model(input_ids=input_ids, attention_mask=attn_mask)
            token_embeddings = out[0]

            if self.device == "cuda":
                input_mask_expanded = attn_mask.unsqueeze(-1).expand(token_embeddings.size()).half()
            else:
                input_mask_expanded = attn_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            pooled = sum_embeddings / sum_mask

            if self.normalize:
                normed = torch.nn.functional.normalize(pooled, p=2, dim=1)
            else:
                normed = pooled

            vec = normed[0].float().cpu().numpy()

        latency_ms = (time.perf_counter_ns() - t0) / 1_000_000.0

        if vec.ndim == 2:
            vec = vec[0]
        return vec.astype(np.float32), latency_ms

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
