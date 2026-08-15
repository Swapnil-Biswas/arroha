"""
indexing/bm25_index.py
----------------------
BM25 lexical index manager with multilingual tokenization.
Provides complementary keyword-matching signals to dense vector search.
"""

from __future__ import annotations

import json
import logging
import pickle
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Optional

from rank_bm25 import BM25Okapi

from app.config import BM25_INDEX_PATH, BM25_METADATA_PATH
from ingestion.models import Chunk

logger = logging.getLogger(__name__)

# Multilingual tokenization regex: captures unicode words across all Indic and Latin scripts
MULTILINGUAL_WORD_RE = re.compile(r"[\w\u0900-\u0D7F]+", re.UNICODE)


def tokenize_multilingual(text: str) -> list[str]:
    """
    Multilingual tokenizer for BM25.
    Normalizes Unicode, extracts words across Indic scripts & Latin, lowercases.
    """
    if not text:
        return []
    normalized = unicodedata.normalize("NFC", text.lower())
    tokens = MULTILINGUAL_WORD_RE.findall(normalized)
    return [t for t in tokens if len(t) > 0]


class BM25IndexManager:
    """
    Manages BM25Okapi lexical index and metadata storage.
    """

    def __init__(
        self,
        index_path: Path = BM25_INDEX_PATH,
        metadata_path: Path = BM25_METADATA_PATH,
    ) -> None:
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.model: Optional[BM25Okapi] = None
        self.metadata: list[dict[str, Any]] = []

    @property
    def is_ready(self) -> bool:
        return self.model is not None and len(self.metadata) > 0

    @property
    def count(self) -> int:
        return len(self.metadata) if self.metadata else 0

    def build_index(self, chunks: list[Chunk]) -> None:
        """
        Tokenize chunk texts and build the in-memory BM25Okapi index.
        """
        if not chunks:
            logger.warning("No chunks provided to build BM25 index.")
            return

        logger.info("Tokenizing %d chunks for BM25 index...", len(chunks))
        tokenized_corpus: list[list[str]] = []
        metadata_list: list[dict[str, Any]] = []

        for chunk in chunks:
            tokens = tokenize_multilingual(chunk.text)
            tokenized_corpus.append(tokens)
            metadata_list.append(chunk.model_dump())

        self.model = BM25Okapi(tokenized_corpus)
        self.metadata = metadata_list
        logger.info("BM25 index built with %d documents.", len(self.metadata))

    def save(self) -> None:
        """Persist BM25 model and metadata to disk."""
        if self.model is None:
            raise RuntimeError("Cannot save uninitialized BM25 index.")

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.model, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            for item in self.metadata:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info("Saved BM25 index to %s and metadata to %s", self.index_path, self.metadata_path)

    def load(self) -> bool:
        """Load BM25 model and metadata from disk into memory."""
        if not self.index_path.exists() or not self.metadata_path.exists():
            logger.warning("BM25 index or metadata file not found at %s", self.index_path)
            return False

        logger.info("Loading BM25 index from %s...", self.index_path)
        with open(self.index_path, "rb") as f:
            self.model = pickle.load(f)

        self.metadata = []
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.metadata.append(json.loads(line))

        logger.info("Loaded BM25 index with %d documents.", len(self.metadata))
        return True

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> tuple[list[tuple[dict[str, Any], float]], float]:
        """
        Search BM25 index for a query.
        Returns (list_of_(chunk_dict, raw_bm25_score), search_latency_ms).
        """
        if not self.is_ready or self.model is None:
            return [], 0.0

        t0 = time.perf_counter_ns()
        query_tokens = tokenize_multilingual(query)

        if not query_tokens:
            latency_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            return [], latency_ms

        scores = self.model.get_scores(query_tokens)
        latency_ms = (time.perf_counter_ns() - t0) / 1_000_000.0

        # Extract top-k indices with non-zero score
        indexed_scores = [(idx, score) for idx, score in enumerate(scores) if score > 0]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = indexed_scores[:top_k]

        results: list[tuple[dict[str, Any], float]] = []
        for idx, score in top_indices:
            results.append((self.metadata[idx], float(score)))

        return results, latency_ms
