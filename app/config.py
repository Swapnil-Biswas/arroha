"""
app/config.py
-------------
Centralized configuration management for HH Goa 2026 Voice-Enabled RAG.
Reads from environment variables / .env file with sensible production defaults.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# ---------------------------------------------------------------------------
# Base Directories
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDEX_DIR = BASE_DIR / "indexes"
LOGS_DIR = BASE_DIR / "logs"

# Ensure all primary runtime directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, INDEX_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset Configuration
# ---------------------------------------------------------------------------
DATASET_ID: str = os.getenv("DATASET_ID", "ai4bharat/MSMARCO-XI")
DATASET_CONFIG: str = os.getenv("DATASET_CONFIG", "default")
SUPPORTED_LANGUAGES: list[str] = [
    "asm", "ben", "guj", "hin", "kan", "mal",
    "mar", "nep", "ori", "pan", "san", "tam", "tel", "urd"
]

# ---------------------------------------------------------------------------
# Chunking Configuration
# ---------------------------------------------------------------------------
ChunkingStrategy = Literal["fixed", "sentence", "passage", "recursive"]
CHUNKING_STRATEGY: ChunkingStrategy = os.getenv("CHUNKING_STRATEGY", "sentence")  # type: ignore
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "300"))           # characters / tokens
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))       # overlap characters

# ---------------------------------------------------------------------------
# Embedding Model Configuration
# ---------------------------------------------------------------------------
# Using high-throughput multilingual model with 384 dims (fast on GPU & CPU)
EMBEDDING_MODEL_ID: str = os.getenv(
    "EMBEDDING_MODEL_ID",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "384"))
EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cuda" if os.getenv("USE_CUDA", "1") == "1" else "cpu")
NORMALIZE_EMBEDDINGS: bool = os.getenv("NORMALIZE_EMBEDDINGS", "true").lower() in ("true", "1", "yes")

# ---------------------------------------------------------------------------
# Index & Retrieval Configuration
# ---------------------------------------------------------------------------
FAISS_INDEX_PATH: Path = INDEX_DIR / os.getenv("FAISS_INDEX_FILE", "vector.faiss")
FAISS_METADATA_PATH: Path = INDEX_DIR / os.getenv("FAISS_METADATA_FILE", "vector_meta.jsonl")
BM25_INDEX_PATH: Path = INDEX_DIR / os.getenv("BM25_INDEX_FILE", "bm25.pkl")
BM25_METADATA_PATH: Path = INDEX_DIR / os.getenv("BM25_METADATA_FILE", "bm25_meta.jsonl")

# Retrieval Hyperparameters
RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
DENSE_WEIGHT: float = float(os.getenv("DENSE_WEIGHT", "0.6"))
BM25_WEIGHT: float = float(os.getenv("BM25_WEIGHT", "0.4"))
ENABLE_RERANKER: bool = os.getenv("ENABLE_RERANKER", "false").lower() in ("true", "1", "yes")
MIN_RETRIEVAL_SCORE: float = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.25"))

# ---------------------------------------------------------------------------
# LLM Generation Configuration (Qwen3 4B 2507 Q4_K_M baseline)
# ---------------------------------------------------------------------------
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "lm_studio")  # lm_studio, openai_compat, mock
LLM_ENDPOINT: str = os.getenv("LLM_ENDPOINT", "http://localhost:1234/v1")
LLM_API_KEY: str = os.getenv("LLM_API_KEY", "lm-studio")
LLM_MODEL_ID: str = os.getenv("LLM_MODEL_ID", "qwen/qwen3-4b-2507")
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "150"))
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "8.0"))

# ---------------------------------------------------------------------------
# Speech-To-Text Configuration
# ---------------------------------------------------------------------------
STT_PROVIDER: str = os.getenv("STT_PROVIDER", "faster_whisper")  # faster_whisper, mock
STT_MODEL_SIZE: str = os.getenv("STT_MODEL_SIZE", "tiny")        # tiny, base (optimized for <50ms latency)
STT_DEVICE: str = os.getenv("STT_DEVICE", "cuda" if os.getenv("USE_CUDA", "1") == "1" else "cpu")

# ---------------------------------------------------------------------------
# Latency & Guardrails Thresholds
# ---------------------------------------------------------------------------
LATENCY_BUDGET_MS: float = float(os.getenv("LATENCY_BUDGET_MS", "200.0"))
STRETCH_LATENCY_BUDGET_MS: float = float(os.getenv("STRETCH_LATENCY_BUDGET_MS", "150.0"))
GROUNDING_SIMILARITY_THRESHOLD: float = float(os.getenv("GROUNDING_SIMILARITY_THRESHOLD", "0.35"))

# ---------------------------------------------------------------------------
# FastAPI Server
# ---------------------------------------------------------------------------
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
DEBUG: bool = os.getenv("DEBUG", "true").lower() in ("true", "1", "yes")
