"""
ingestion/build_index.py
------------------------
Orchestrates end-to-end index construction:
1. Ingestion: Reads dataset records (sample or full Parquet shards).
2. Preprocessing: Normalizes multilingual text and extracts canonical Documents.
3. Chunking: Applies chosen strategy (sentence, fixed, passage, recursive).
4. Embedding: Generates dense multilingual vector embeddings.
5. Indexing: Builds and persists FAISS vector index + BM25Okapi lexical index.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKING_STRATEGY,
    DATA_DIR,
    INDEX_DIR,
    SUPPORTED_LANGUAGES,
)
from ingestion.chunking import get_chunker
from ingestion.download import (
    create_sample_multilingual_corpus,
    download_language_shard,
    stream_records_from_parquet,
)
from ingestion.models import Chunk, Document
from ingestion.preprocess import batch_preprocess_records
from indexing.bm25_index import BM25IndexManager
from indexing.embeddings import MultilingualEmbedder
from indexing.faiss_index import FAISSIndexManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_index")


def build_pipeline_indexes(
    use_sample: bool = True,
    language_shards: Optional[list[str]] = None,
    max_records_per_shard: int = 500,
    chunking_strategy: str = CHUNKING_STRATEGY,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> tuple[int, int, int]:
    """
    Build and save both FAISS vector index and BM25 lexical index.
    Returns (num_records, num_documents, num_chunks).
    """
    t_start = time.perf_counter()
    logger.info("=== Starting Index Construction ===")
    logger.info("Strategy: %s | Chunk Size: %d | Overlap: %d", chunking_strategy, chunk_size, chunk_overlap)

    # 1. Acquire records
    records = []
    if use_sample or not language_shards:
        logger.info("Generating curated multilingual development dataset...")
        records = create_sample_multilingual_corpus()
    else:
        for lang in language_shards:
            try:
                parquet_path = download_language_shard(lang=lang)
                logger.info("Streaming up to %d records from %s...", max_records_per_shard, parquet_path.name)
                for rec in stream_records_from_parquet(parquet_path, max_records=max_records_per_shard):
                    records.append(rec)
            except Exception as exc:
                logger.error("Failed to load shard for language '%s': %s", lang, exc)

    logger.info("Step 1 Complete: Acquired %d dataset records.", len(records))

    # 2. Preprocess records into Documents
    logger.info("Step 2: Preprocessing and extracting canonical documents...")
    documents: list[Document] = list(batch_preprocess_records(records, include_translated=True, include_english=True))
    logger.info("Extracted %d canonical documents (clean text, no gold answers).", len(documents))

    # 3. Chunk documents
    logger.info("Step 3: Chunking documents with strategy '%s'...", chunking_strategy)
    chunker = get_chunker(strategy=chunking_strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[Chunk] = list(chunker.chunk_documents(documents))
    logger.info("Generated %d searchable chunks.", len(chunks))

    if not chunks:
        logger.error("No chunks generated! Aborting index build.")
        return 0, 0, 0

    # 4. Generate dense embeddings
    logger.info("Step 4: Generating multilingual embeddings for %d chunks...", len(chunks))
    t_embed_start = time.perf_counter()
    embedder = MultilingualEmbedder.get_instance()
    chunk_texts = [c.text for c in chunks]
    embeddings = embedder.embed_documents(chunk_texts, show_progress=True)
    embed_time = time.perf_counter() - t_embed_start
    logger.info("Embeddings complete in %.2f s (%.1f chunks/sec). Shape: %s", embed_time, len(chunks) / max(embed_time, 0.001), embeddings.shape)

    # 5. Build & Save FAISS Index
    logger.info("Step 5a: Building FAISS vector index...")
    faiss_mgr = FAISSIndexManager()
    faiss_mgr.build_index(embeddings=embeddings, chunks=chunks, index_type="FlatIP")
    faiss_mgr.save()

    # 6. Build & Save BM25 Index
    logger.info("Step 5b: Building BM25 lexical index...")
    bm25_mgr = BM25IndexManager()
    bm25_mgr.build_index(chunks=chunks)
    bm25_mgr.save()

    total_time = time.perf_counter() - t_start
    logger.info("=== Index Construction Complete in %.2f seconds ===", total_time)
    logger.info("Summary: %d Records -> %d Documents -> %d Chunks indexed.", len(records), len(documents), len(chunks))

    return len(records), len(documents), len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS and BM25 indexes for HH Goa Voice RAG.")
    parser.add_argument("--sample", action="store_true", default=True, help="Use curated multilingual sample corpus")
    parser.add_argument("--full", action="store_true", help="Download and index full language shards")
    parser.add_argument("--languages", nargs="+", default=["hin", "ben", "tam"], help="Language shards to index (e.g. hin ben tam)")
    parser.add_argument("--max-records", type=int, default=200, help="Max records per language shard")
    parser.add_argument("--strategy", type=str, default=CHUNKING_STRATEGY, choices=["fixed", "sentence", "passage", "recursive"])
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)

    args = parser.parse_args()

    use_sample = not args.full

    build_pipeline_indexes(
        use_sample=use_sample,
        language_shards=args.languages if args.full else None,
        max_records_per_shard=args.max_records,
        chunking_strategy=args.strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )


if __name__ == "__main__":
    main()
