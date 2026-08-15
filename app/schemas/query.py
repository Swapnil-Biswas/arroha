"""
app/schemas/query.py
--------------------
Pydantic models for incoming user queries (text and voice).
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Text-based RAG query request."""
    query: str = Field(..., min_length=1, max_length=1000, description="User query text")
    language: Optional[str] = Field(None, description="Optional ISO/Indic language code (e.g. 'hin', 'ben', 'eng')")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Override default top_k candidate retrieval")
    dense_weight: Optional[float] = Field(None, ge=0.0, le=1.0, description="Dense retrieval weight in hybrid fusion")
    bm25_weight: Optional[float] = Field(None, ge=0.0, le=1.0, description="Sparse BM25 weight in hybrid fusion")
    include_debug: bool = Field(False, description="Whether to include granular debugging details in response")


class VoiceQueryRequest(BaseModel):
    """Voice query with raw base64 encoded audio or metadata."""
    audio_base64: Optional[str] = Field(None, description="Base64 encoded audio payload (WAV/MP3/PCM)")
    audio_format: str = Field("wav", description="Audio format (wav, mp3, ogg, webm)")
    sample_rate: int = Field(16000, description="Audio sample rate in Hz")
    language_hint: Optional[str] = Field(None, description="Optional language hint for STT")
    top_k: Optional[int] = Field(None, ge=1, le=20)
    include_debug: bool = Field(False)


class BenchmarkQuery(BaseModel):
    """Query object used in latency & retrieval benchmarking."""
    query_id: str | int
    query_text: str
    language: str
    expected_passage_ids: list[str | int] = Field(default_factory=list)
    gold_answer: Optional[str] = None
