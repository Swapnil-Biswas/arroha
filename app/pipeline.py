"""
app/pipeline.py
---------------
End-to-End Multilingual Voice-Enabled RAG Pipeline Orchestrator.
Coordinates: Voice -> STT -> Input Guardrails -> Hybrid Retrieval ->
LLM Generation -> Grounding Verification -> Output Guardrails.

Instruments every stage with nanosecond monotonic timers.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Optional

from app.config import (
    LATENCY_BUDGET_MS,
    MIN_RETRIEVAL_SCORE,
    RETRIEVAL_TOP_K,
    STRETCH_LATENCY_BUDGET_MS,
    TTS_BACKEND,
    TTS_BUFFER_MODE,
    TTS_SAMPLE_RATE,
)
from app.generation.llm import LLMGenerator
from app.generation.prompts import build_rag_prompt
from app.guardrails.validator import GuardrailsValidator
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import Reranker
from app.schemas.query import QueryRequest, VoiceQueryRequest
from app.schemas.response import (
    GroundingResult,
    LatencyBreakdown,
    RAGResponse,
    SourceDocument,
    VoiceStreamChunk,
)
from app.voice.language_router import LanguageRouter
from app.voice.pipeline import StreamingVoicePipeline
from app.voice.stt import SpeechToTextEngine
from app.voice.tts_backend import TTSBackend, create_tts_backend

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Core RAG Pipeline coordinating all components.
    Supports both synchronous text RAG and concurrent streaming voice RAG.
    """

    def __init__(
        self,
        hybrid_retriever: Optional[HybridRetriever] = None,
        llm_generator: Optional[LLMGenerator] = None,
        guardrails_validator: Optional[GuardrailsValidator] = None,
        stt_engine: Optional[SpeechToTextEngine] = None,
        reranker: Optional[Reranker] = None,
        tts_backend: Optional[TTSBackend] = None,
        streaming_voice_pipeline: Optional[StreamingVoicePipeline] = None,
    ) -> None:
        self.hybrid_retriever = hybrid_retriever or HybridRetriever()
        self.llm_generator = llm_generator or LLMGenerator()
        self.guardrails = guardrails_validator or GuardrailsValidator()
        self.stt = stt_engine or SpeechToTextEngine()
        self.reranker = reranker or Reranker()
        self.tts_backend = tts_backend or create_tts_backend(TTS_BACKEND, sample_rate=TTS_SAMPLE_RATE)
        self.voice_pipeline = streaming_voice_pipeline or StreamingVoicePipeline(
            tts_backend=self.tts_backend,
            buffer_mode=TTS_BUFFER_MODE,
        )

    def process_query(self, request: QueryRequest) -> RAGResponse:
        """
        Execute full RAG pipeline for a text query.
        """
        t_pipeline_start = time.perf_counter_ns()
        request_id = str(uuid.uuid4())[:8]

        latency = LatencyBreakdown()

        # 1. Input Guardrails
        t_in_start = time.perf_counter_ns()
        is_valid, cleaned_query, detected_script, error_reason, in_latency = self.guardrails.validate_input(
            request.query, language_hint=request.language
        )
        latency.input_guardrails_ms = round(in_latency, 2)

        if not is_valid:
            total_ms = (time.perf_counter_ns() - t_pipeline_start) / 1_000_000.0
            latency.total_ms = round(total_ms, 2)
            latency.target_achieved_200ms = total_ms <= LATENCY_BUDGET_MS
            latency.stretch_achieved_150ms = total_ms <= STRETCH_LATENCY_BUDGET_MS

            return RAGResponse(
                query=request.query,
                detected_language=detected_script,
                answer=error_reason or "Invalid query provided.",
                is_refusal=True,
                grounding=GroundingResult(
                    is_grounded=False,
                    grounding_score=0.0,
                    refusal_triggered=True,
                    refusal_reason=error_reason,
                ),
                sources=[],
                latency=latency,
                request_id=request_id,
            )

        # 2. Hybrid Retrieval (Dense Vector + Sparse BM25 + Weighted Fusion)
        top_k = request.top_k or RETRIEVAL_TOP_K
        sources, ret_latencies = self.hybrid_retriever.search(
            query=cleaned_query,
            top_k=top_k,
            dense_weight=request.dense_weight,
            bm25_weight=request.bm25_weight,
        )

        latency.query_embed_ms = ret_latencies.get("query_embed_ms", 0.0)
        latency.vector_retrieval_ms = ret_latencies.get("vector_retrieval_ms", 0.0)
        latency.bm25_retrieval_ms = ret_latencies.get("bm25_retrieval_ms", 0.0)
        latency.hybrid_fusion_ms = ret_latencies.get("hybrid_fusion_ms", 0.0)

        # 3. Optional Reranking
        sources, rerank_ms = self.reranker.rerank(cleaned_query, sources, top_k=top_k)
        latency.reranker_ms = round(rerank_ms, 2)

        # 4. Prompt Assembly
        t_prompt_start = time.perf_counter_ns()
        system_prompt, user_msg = build_rag_prompt(cleaned_query, sources)
        latency.prompt_construction_ms = round((time.perf_counter_ns() - t_prompt_start) / 1_000_000.0, 2)

        # 5. LLM Generation
        raw_answer, ttft_ms, gen_ms, tok_count, gen_tps, e2e_tps = self.llm_generator.generate_stream(cleaned_query, sources)
        latency.llm_ttft_ms = round(ttft_ms, 2)
        latency.llm_generation_ms = round(gen_ms, 2)

        # 6. Grounding Check
        grounding_res, ground_ms = self.guardrails.check_grounding(cleaned_query, sources, raw_answer)
        latency.grounding_check_ms = round(ground_ms, 2)

        # 7. Output Sanitization
        final_answer, _ = self.guardrails.sanitize_output(raw_answer, is_refusal=grounding_res.refusal_triggered)

        # Total latency computation
        total_ms = (time.perf_counter_ns() - t_pipeline_start) / 1_000_000.0
        latency.total_ms = round(total_ms, 2)
        latency.target_achieved_200ms = total_ms <= LATENCY_BUDGET_MS
        latency.stretch_achieved_150ms = total_ms <= STRETCH_LATENCY_BUDGET_MS

        debug_info = None
        if request.include_debug:
            debug_info = {
                "detected_script": detected_script,
                "candidate_count": len(sources),
                "max_score": sources[0].score if sources else 0.0,
                "system_prompt_len": len(system_prompt),
            }

        return RAGResponse(
            query=cleaned_query,
            detected_language=detected_script,
            answer=final_answer,
            is_refusal=grounding_res.refusal_triggered,
            grounding=grounding_res,
            sources=sources,
            latency=latency,
            request_id=request_id,
            debug_info=debug_info,
        )

    def process_voice_query(self, request: VoiceQueryRequest) -> RAGResponse:
        """
        Execute full Voice-Enabled RAG pipeline: Voice -> STT -> RAG Retrieval -> LLM -> Concurrent TTS.
        """
        t_voice_start = time.perf_counter_ns()

        if not request.audio_base64:
            return RAGResponse(
                query="",
                detected_language="Unknown",
                answer="No audio payload provided.",
                is_refusal=True,
                grounding=GroundingResult(is_grounded=False, refusal_triggered=True, refusal_reason="Empty audio"),
                sources=[],
                latency=LatencyBreakdown(total_ms=0.0),
            )

        # 1. Speech-To-Text
        transcribed_text, detected_lang, stt_ms = self.stt.transcribe(
            audio_data=request.audio_base64,
            language_hint=request.language_hint,
            audio_format=request.audio_format,
        )

        if not transcribed_text:
            return RAGResponse(
                query="",
                detected_language=detected_lang,
                answer="Unable to transcribe audio or audio was silent.",
                is_refusal=True,
                grounding=GroundingResult(is_grounded=False, refusal_triggered=True, refusal_reason="STT failure"),
                sources=[],
                latency=LatencyBreakdown(stt_ms=round(stt_ms, 2), total_ms=round(stt_ms, 2)),
            )

        # 2. Delegate to Text RAG
        text_request = QueryRequest(
            query=transcribed_text,
            language=detected_lang,
            top_k=request.top_k,
            include_debug=request.include_debug,
        )
        response = self.process_query(text_request)

        # 3. If Voice Mode requested, synthesize speech
        if request.mode == "voice":
            v_cfg = LanguageRouter.get_voice_config(detected_lang)
            t_synth_0 = time.perf_counter_ns()
            audio_chunk = self.tts_backend.synthesize_chunk(response.answer, language=detected_lang)
            synth_ms = (time.perf_counter_ns() - t_synth_0) / 1e6
            response.audio_base64 = audio_chunk.audio_base64
            response.audio_format = "wav"
            response.voice_type = v_cfg.get("voice_type", "NATIVE VOICE")
            response.latency.tts_first_chunk_ms = round(synth_ms, 2)
            first_audio_ms = response.latency.llm_ttft_ms + round(synth_ms, 2)
            response.latency.first_audio_latency_ms = round(first_audio_ms, 2)

        # Update STT latency and overall total
        response.latency.stt_ms = round(stt_ms, 2)
        total_voice_ms = (time.perf_counter_ns() - t_voice_start) / 1_000_000.0
        response.latency.total_ms = round(total_voice_ms, 2)
        response.latency.target_achieved_200ms = total_voice_ms <= LATENCY_BUDGET_MS
        response.latency.stretch_achieved_150ms = total_voice_ms <= STRETCH_LATENCY_BUDGET_MS

        return response

    def stream_voice_query(
        self, request: VoiceQueryRequest, session_id: Optional[str] = None
    ) -> Generator[VoiceStreamChunk, None, None]:
        """
        Streaming Voice Pipeline Generator:
        Yields real-time events (status, transcript, token, audio_chunk, done).
        """
        sid = session_id or request.session_id or str(uuid.uuid4())[:8]

        # 1. STT Phase
        yield VoiceStreamChunk(event="status", session_id=sid, text="LISTENING")

        if not request.audio_base64:
            yield VoiceStreamChunk(
                event="error",
                session_id=sid,
                text="No audio payload provided.",
                is_final=True,
            )
            return

        transcribed_text, detected_lang, stt_ms = self.stt.transcribe(
            audio_data=request.audio_base64,
            language_hint=request.language_hint,
            audio_format=request.audio_format,
        )

        if not transcribed_text:
            yield VoiceStreamChunk(
                event="error",
                session_id=sid,
                text="Unable to transcribe audio or audio was silent.",
                is_final=True,
            )
            return

        # Emit transcription to client
        yield VoiceStreamChunk(
            event="transcript",
            session_id=sid,
            text=transcribed_text,
        )

        # 2. Retrieval Phase
        is_valid, cleaned_query, detected_script, error_reason, in_latency = self.guardrails.validate_input(
            transcribed_text, language_hint=detected_lang
        )

        if not is_valid:
            yield VoiceStreamChunk(
                event="error",
                session_id=sid,
                text=error_reason or "Invalid query input.",
                is_final=True,
            )
            return

        top_k = request.top_k or RETRIEVAL_TOP_K
        sources, ret_latencies = self.hybrid_retriever.search(query=cleaned_query, top_k=top_k)
        ret_ms = ret_latencies.get("hybrid_fusion_ms", 15.0)

        # 3. Concurrent LLM + TTS Streaming
        system_prompt, user_msg = build_rag_prompt(cleaned_query, sources)

        def stream_supplier():
            return self.llm_generator.client.chat.completions.create(
                model=self.llm_generator.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=self.llm_generator.max_tokens,
                temperature=self.llm_generator.temperature,
                stream=True,
            )

        for event_chunk in self.voice_pipeline.stream_voice_events(
            query=cleaned_query,
            language=detected_lang or detected_script,
            retrieval_ms=ret_ms,
            llm_stream_generator=stream_supplier,
            session_id=sid,
        ):
            if event_chunk.latency:
                event_chunk.latency.stt_ms = round(stt_ms, 2)
            yield event_chunk

    def interrupt(self, session_id: str) -> bool:
        """Interrupt active voice stream for session_id."""
        return self.voice_pipeline.interrupt_session(session_id)
