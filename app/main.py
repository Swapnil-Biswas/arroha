"""
app/main.py
-----------
FastAPI REST API for HH Goa 2026 Voice-Enabled Multilingual RAG.
Exposes /query, /voice, /health, /metrics, and serves the static demo UI.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.config import (
    API_HOST,
    API_PORT,
    DEBUG,
    EMBEDDING_MODEL_ID,
    LATENCY_BUDGET_MS,
    LLM_MODEL_ID,
    TTS_BACKEND,
)
from app.pipeline import RAGPipeline
from app.schemas.query import QueryRequest, VoiceQueryRequest
from app.schemas.response import HealthResponse, RAGResponse, VoiceStreamChunk

# Setup logging
logging.basicConfig(
    level=logging.INFO if not DEBUG else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("api")

app = FastAPI(
    title="HH Goa 2026: Multilingual Voice RAG API",
    description="Low-latency voice-enabled RAG pipeline (<200ms) with concurrent local streaming speech synthesis.",
    version="1.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline instance
pipeline = RAGPipeline()


@app.on_event("startup")
def startup_event():
    """Pre-warm retriever to ensure sub-50ms latency on very first cold query."""
    logger.info("Pre-warming hybrid retriever...")
    try:
        pipeline.hybrid_retriever.search("warmup query", top_k=1)
        logger.info("Pre-warming complete.")
    except Exception as exc:
        logger.warning(f"Pre-warming failed (index might not exist yet): {exc}")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Check pipeline health, index status, and hardware acceleration."""
    is_vec_ready = pipeline.hybrid_retriever.vector_retriever.is_ready
    is_bm25_ready = pipeline.hybrid_retriever.bm25_retriever.is_ready
    total_docs = pipeline.hybrid_retriever.vector_retriever.index_manager.count

    return HealthResponse(
        status="ok" if (is_vec_ready or is_bm25_ready) else "indexes_empty",
        version="1.1.0",
        model_id=LLM_MODEL_ID,
        embedding_model=EMBEDDING_MODEL_ID,
        vector_index_ready=is_vec_ready,
        bm25_index_ready=is_bm25_ready,
        total_indexed_documents=total_docs,
        gpu_available=torch.cuda.is_available(),
        tts_backend=TTS_BACKEND,
        target_latency_budget_ms=LATENCY_BUDGET_MS,
    )


@app.post("/query", response_model=RAGResponse)
def process_text_query(request: QueryRequest) -> RAGResponse:
    """Execute full multilingual text RAG pipeline (mode=text)."""
    try:
        return pipeline.process_query(request)
    except Exception as exc:
        logger.error("Error processing query: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing the query.",
        )


@app.post("/voice", response_model=RAGResponse)
def process_voice_query(request: VoiceQueryRequest) -> RAGResponse:
    """Execute voice RAG pipeline with base64 encoded audio payload (mode=voice)."""
    try:
        return pipeline.process_voice_query(request)
    except Exception as exc:
        logger.error("Error processing voice query: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing voice query.",
        )


@app.post("/voice/stream")
def stream_voice_query(request: VoiceQueryRequest) -> StreamingResponse:
    """
    Execute streaming voice RAG pipeline via Server-Sent Events (SSE).
    Streams delta text tokens and synthesized audio frames concurrently as LLM generates.
    """
    def event_generator():
        try:
            for event_chunk in pipeline.stream_voice_query(request):
                payload = event_chunk.model_dump_json()
                yield f"event: {event_chunk.event}\ndata: {payload}\n\n"
        except Exception as exc:
            logger.error("Streaming error: %s", exc, exc_info=True)
            err_chunk = VoiceStreamChunk(event="error", text=str(exc), is_final=True)
            yield f"event: error\ndata: {err_chunk.model_dump_json()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/voice/interrupt")
def interrupt_voice_session(session_id: str = "default") -> dict[str, Any]:
    """
    Barge-in / Interruption endpoint: Immediately stops speech synthesis & audio playback for active session.
    """
    interrupted = pipeline.interrupt(session_id)
    return {
        "session_id": session_id,
        "interrupted": interrupted,
        "status": "cancelled" if interrupted else "no_active_session",
    }


@app.post("/voice/upload", response_model=RAGResponse)
async def process_voice_upload(file: UploadFile = File(...)) -> RAGResponse:
    """Execute voice RAG pipeline directly from uploaded audio file (WAV/MP3/WEBM)."""
    try:
        audio_bytes = await file.read()
        b64_audio = base64.b64encode(audio_bytes).decode("utf-8")
        req = VoiceQueryRequest(
            audio_base64=b64_audio,
            audio_format=file.filename.split(".")[-1] if file.filename else "wav",
            mode="voice",
        )
        return pipeline.process_voice_query(req)
    except Exception as exc:
        logger.error("Error processing uploaded audio file: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing audio file upload.",
        )


@app.get("/metrics")
def get_metrics() -> dict[str, Any]:
    """Retrieve system configuration and latency threshold metrics."""
    return {
        "target_latency_ms": LATENCY_BUDGET_MS,
        "stretch_latency_ms": 150.0,
        "indexed_chunks": pipeline.hybrid_retriever.vector_retriever.index_manager.count,
        "dense_weight": pipeline.hybrid_retriever.dense_weight,
        "bm25_weight": pipeline.hybrid_retriever.bm25_weight,
        "embedding_model": EMBEDDING_MODEL_ID,
        "llm_model": LLM_MODEL_ID,
        "tts_backend": TTS_BACKEND,
        "cuda_enabled": torch.cuda.is_available(),
    }


# Static files mount for Web Demo UI
static_dir = Path(__file__).resolve().parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    def serve_index() -> FileResponse:
        return FileResponse(str(static_dir / "index.html"))
