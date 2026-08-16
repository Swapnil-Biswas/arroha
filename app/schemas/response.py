"""
app/schemas/response.py
-----------------------
Pydantic models for structured output, retrieval sources,
latency breakdowns, and health status.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
    """A retrieved passage/document with scores and metadata."""
    doc_id: str
    text: str
    language: str
    score: float = Field(..., description="Fused hybrid retrieval score [0.0 - 1.0]")
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    query_id: Optional[str | int] = None
    passage_id: Optional[int] = None
    is_selected: Optional[int] = None  # Evaluation metadata only


class LatencyBreakdown(BaseModel):
    """Detailed millisecond measurements across each pipeline stage."""
    stt_ms: float = 0.0
    input_guardrails_ms: float = 0.0
    query_embed_ms: float = 0.0
    bm25_retrieval_ms: float = 0.0
    vector_retrieval_ms: float = 0.0
    hybrid_fusion_ms: float = 0.0
    reranker_ms: float = 0.0
    prompt_construction_ms: float = 0.0
    llm_ttft_ms: float = 0.0
    llm_generation_ms: float = 0.0
    tts_first_chunk_ms: float = 0.0
    first_audio_latency_ms: float = 0.0
    grounding_check_ms: float = 0.0
    total_ms: float = 0.0

    target_achieved_200ms: bool = False
    stretch_achieved_150ms: bool = False


class GroundingResult(BaseModel):
    """Grounding & hallucination check outcome."""
    is_grounded: bool = True
    grounding_score: float = 1.0
    refusal_triggered: bool = False
    refusal_reason: Optional[str] = None


class VoiceStreamChunk(BaseModel):
    """Event-stream packet for real-time speech and token transport."""
    event: str = Field(..., description="'status', 'transcript', 'token', 'audio_chunk', 'metrics', 'done', 'error'")
    session_id: Optional[str] = None
    text: Optional[str] = None
    delta: Optional[str] = None
    audio_base64: Optional[str] = None
    chunk_index: Optional[int] = None
    audio_duration_ms: Optional[float] = None
    synthesis_latency_ms: Optional[float] = None
    is_final: bool = False
    latency: Optional[LatencyBreakdown] = None


class RAGResponse(BaseModel):
    """Standard structured output for RAG queries."""
    query: str
    detected_language: str
    answer: str
    is_refusal: bool = False
    grounding: GroundingResult
    sources: list[SourceDocument] = Field(default_factory=list)
    latency: LatencyBreakdown
    request_id: Optional[str] = None
    debug_info: Optional[dict[str, Any]] = None
    audio_base64: Optional[str] = None
    audio_format: Optional[str] = None
    voice_type: Optional[str] = None


class HealthResponse(BaseModel):
    """System health check and index readiness status."""
    status: str = "ok"
    version: str = "1.0.0"
    model_id: str
    embedding_model: str
    vector_index_ready: bool
    bm25_index_ready: bool
    total_indexed_documents: int
    gpu_available: bool
    tts_backend: str = "local_onnx"
    target_latency_budget_ms: float = 200.0
