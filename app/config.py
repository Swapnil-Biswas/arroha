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

import torch

# ---------------------------------------------------------------------------
# Embedding Model Configuration
# ---------------------------------------------------------------------------
# Using high-throughput multilingual model with 384 dims (fast on GPU & CPU & MPS)
EMBEDDING_MODEL_ID: str = os.getenv(
    "EMBEDDING_MODEL_ID",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "384"))
EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))

def _detect_best_device() -> str:
    if os.getenv("EMBEDDING_DEVICE"):
        return os.environ["EMBEDDING_DEVICE"]
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

EMBEDDING_DEVICE: str = _detect_best_device()
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
DENSE_WEIGHT: float = float(os.getenv("DENSE_WEIGHT", "0.45"))
BM25_WEIGHT: float = float(os.getenv("BM25_WEIGHT", "0.55"))
ENABLE_RERANKER: bool = os.getenv("ENABLE_RERANKER", "false").lower() in ("true", "1", "yes")
MIN_RETRIEVAL_SCORE: float = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.25"))
PARALLEL_HYBRID_SEARCH: bool = os.getenv("PARALLEL_HYBRID_SEARCH", "true").lower() in ("true", "1", "yes")

# Query Caching Configuration
ENABLE_QUERY_CACHE: bool = os.getenv("ENABLE_QUERY_CACHE", "true").lower() in ("true", "1", "yes")
QUERY_CACHE_SIZE: int = int(os.getenv("QUERY_CACHE_SIZE", "1000"))

# ---------------------------------------------------------------------------
# LLM Generation Configuration (Qwen2.5-1.5B-Instruct Q4_K_M validated)
# ---------------------------------------------------------------------------
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "fast_extractive")  # fast_extractive (<50ms), openai_compat (llama-server), lm_studio, mock
LLM_ENDPOINT: str = os.getenv("LLM_ENDPOINT", "http://127.0.0.1:8080/v1")
LLM_API_KEY: str = os.getenv("LLM_API_KEY", "dummy")
LLM_MODEL_ID: str = os.getenv("LLM_MODEL_ID", "qwen2.5-1.5b-instruct")
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "64"))
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "8.0"))

# Caching & Sub-50ms Optimization Configuration
ENABLE_RAG_CACHE: bool = os.getenv("ENABLE_RAG_CACHE", "true").lower() in ("true", "1", "yes")
CACHE_MAX_SIZE: int = int(os.getenv("CACHE_MAX_SIZE", "4096"))
ENABLE_FAST_PATH_SYNTHESIS: bool = os.getenv("ENABLE_FAST_PATH_SYNTHESIS", "true").lower() in ("true", "1", "yes")

# ---------------------------------------------------------------------------
# Text-To-Speech (TTS) Configuration
# ---------------------------------------------------------------------------
TTS_BACKEND: str = os.getenv("TTS_BACKEND", "local_onnx")  # local_onnx, edge_tts, mock
TTS_BUFFER_MODE: str = os.getenv("TTS_BUFFER_MODE", "adaptive")  # adaptive, tok3_min, sentence
TTS_SAMPLE_RATE: int = int(os.getenv("TTS_SAMPLE_RATE", "24000"))

# ---------------------------------------------------------------------------
# Speech-To-Text Configuration
# ---------------------------------------------------------------------------
STT_BACKEND: str = os.getenv("STT_BACKEND", "local")              # local, sarvam
SARVAM_API_KEY: str = os.getenv("SARVAM_API_KEY", "")
SARVAM_STT_ENDPOINT: str = os.getenv("SARVAM_STT_ENDPOINT", "https://api.sarvam.ai/speech-to-text")
SARVAM_STT_MODEL: str = os.getenv("SARVAM_STT_MODEL", "saaras:v4")  # saaras:v4, saaras:v3, saarika:v2.5
SARVAM_TIMEOUT_SECONDS: float = float(os.getenv("SARVAM_TIMEOUT_SECONDS", "6.0"))
SARVAM_FALLBACK_TO_LOCAL: bool = os.getenv("SARVAM_FALLBACK_TO_LOCAL", "true").lower() in ("true", "1", "yes")

STT_PROVIDER: str = os.getenv("STT_PROVIDER", "faster_whisper")  # faster_whisper, mock
STT_MODEL_SIZE: str = os.getenv("STT_MODEL_SIZE", "tiny")        # tiny, base (optimized for <50ms latency)
STT_DEVICE: str = _detect_best_device()

# ---------------------------------------------------------------------------
# Latency & Guardrails Thresholds (<50ms Target Budget)
# ---------------------------------------------------------------------------
LATENCY_BUDGET_MS: float = float(os.getenv("LATENCY_BUDGET_MS", "50.0"))
STRETCH_LATENCY_BUDGET_MS: float = float(os.getenv("STRETCH_LATENCY_BUDGET_MS", "30.0"))
GROUNDING_SIMILARITY_THRESHOLD: float = float(os.getenv("GROUNDING_SIMILARITY_THRESHOLD", "0.35"))

# ---------------------------------------------------------------------------
# FastAPI Server
# ---------------------------------------------------------------------------
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
DEBUG: bool = os.getenv("DEBUG", "true").lower() in ("true", "1", "yes")

