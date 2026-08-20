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
import re
import threading
import time
import uuid
from collections import OrderedDict
from typing import Generator, Optional

from app.cache import RAGQueryCache
from app.config import (
    ENABLE_QUERY_CACHE,
    LATENCY_BUDGET_MS,
    MIN_RETRIEVAL_SCORE,
    QUERY_CACHE_SIZE,
    RETRIEVAL_TOP_K,
    STRETCH_LATENCY_BUDGET_MS,
    TTS_BACKEND,
    TTS_BUFFER_MODE,
    TTS_SAMPLE_RATE,
)
from app.generation.llm import LLMGenerator
from app.generation.prompts import build_rag_prompt, resolve_query_language
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
        query_cache: Optional[RAGQueryCache] = None,
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
        self._response_cache: OrderedDict[str, RAGResponse] = OrderedDict()
        self._response_cache_lock = threading.Lock()
        self.query_cache = query_cache or RAGQueryCache(capacity=QUERY_CACHE_SIZE)

    def process_query(self, request: QueryRequest) -> RAGResponse:
        """
        Execute full RAG pipeline for a text query with sub-50ms target optimization.
        """
        t_pipeline_start = time.perf_counter_ns()
        request_id = str(uuid.uuid4())[:8]

        # Check full RAG response cache
        norm_q = request.query.strip().lower()
        cache_key = f"{norm_q}_{request.language}_{request.top_k}"
        from app.config import ENABLE_RAG_CACHE, LATENCY_BUDGET_MS, STRETCH_LATENCY_BUDGET_MS

        if ENABLE_RAG_CACHE:
            with self._response_cache_lock:
                if cache_key in self._response_cache:
                    cached_resp = self._response_cache[cache_key]
                    self._response_cache.move_to_end(cache_key)
                    hit_total_ms = round((time.perf_counter_ns() - t_pipeline_start) / 1_000_000.0, 2)
                    
                    # Clone response with fresh timings and request ID
                    cached_latency = LatencyBreakdown(
                        input_guardrails_ms=0.01,
                        query_embed_ms=0.0,
                        bm25_retrieval_ms=0.0,
                        vector_retrieval_ms=0.0,
                        hybrid_fusion_ms=0.01,
                        reranker_ms=0.0,
                        prompt_construction_ms=0.01,
                        llm_ttft_ms=0.0,
                        llm_generation_ms=0.01,
                        grounding_check_ms=0.01,
                        total_ms=hit_total_ms,
                        target_achieved_50ms=hit_total_ms <= LATENCY_BUDGET_MS,
                        stretch_achieved_30ms=hit_total_ms <= STRETCH_LATENCY_BUDGET_MS,
                        target_achieved_200ms=True,
                        stretch_achieved_150ms=True,
                    )
                    return RAGResponse(
                        query=cached_resp.query,
                        detected_language=cached_resp.detected_language,
                        answer=cached_resp.answer,
                        is_refusal=cached_resp.is_refusal,
                        grounding=cached_resp.grounding,
                        sources=cached_resp.sources,
                        latency=cached_latency,
                        request_id=request_id,
                        debug_info=cached_resp.debug_info,
                    )

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
            latency.target_achieved_50ms = total_ms <= LATENCY_BUDGET_MS
            latency.stretch_achieved_30ms = total_ms <= STRETCH_LATENCY_BUDGET_MS
            latency.target_achieved_200ms = total_ms <= 200.0
            latency.stretch_achieved_150ms = total_ms <= 150.0

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

        # Check conversational queries
        norm_q = re.sub(r"[^\w\s]", "", cleaned_query.lower()).strip()
        conversational_responses = {
            "hello": "Hello! I am ARROHA, your real-time multilingual voice assistant. How can I help you today?",
            "hi": "Hi there! How can I assist you with your queries today?",
            "hey": "Hey! Ask me anything in English or any of the 14 Indic languages.",
            "who are you": "I am ARROHA, an ultra low-latency multilingual voice-enabled RAG assistant built for Hacker House Goa.",
            "what is your name": "I am ARROHA, a high-speed multilingual voice RAG system.",
            "what can you do": "I can answer factual questions with grounded evidence across English and 14 Indic languages in under 50 milliseconds.",
            "tell me a joke": "Why do programmers prefer dark mode? Because light attracts bugs!",
            "how are you": "I'm running fast and ready to help! What question would you like to explore?",
        }
        if norm_q in conversational_responses:
            conv_ans = conversational_responses[norm_q]
            total_ms = (time.perf_counter_ns() - t_pipeline_start) / 1_000_000.0
            latency.total_ms = round(total_ms, 2)
            latency.llm_ttft_ms = 0.1
            latency.llm_generation_ms = 0.1
            return RAGResponse(
                query=request.query,
                detected_language=detected_script,
                answer=conv_ans,
                is_refusal=False,
                grounding=GroundingResult(is_grounded=True, grounding_score=1.0, refusal_triggered=False),
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
            target_language=request.language,
        )

        latency.query_embed_ms = ret_latencies.get("query_embed_ms", 0.0)
        latency.vector_retrieval_ms = ret_latencies.get("vector_retrieval_ms", 0.0)
        latency.bm25_retrieval_ms = ret_latencies.get("bm25_retrieval_ms", 0.0)
        latency.hybrid_fusion_ms = ret_latencies.get("hybrid_fusion_ms", 0.0)

        # Resolve language for prompt and localized refusal
        lang_code, _, _ = resolve_query_language(cleaned_query, language_hint=request.language)
        from app.guardrails.grounding import LOCALIZED_REFUSALS, MIN_RETRIEVAL_SCORE

        # Retrieval Relevance Gating: If no sources or scores below relevance gate, trigger immediate refusal
        max_source_score = max((s.score for s in sources), default=0.0)
        max_dense_score = max((getattr(s, "dense_score", s.score) or 0.0 for s in sources), default=0.0)

        # Check Query-Context Subject Entity Alignment:
        is_aligned, align_score, align_reason = self.guardrails.grounding_checker.check_query_context_alignment(cleaned_query, sources)

        if not sources or max_source_score < MIN_RETRIEVAL_SCORE or max_dense_score < 0.38 or not is_aligned:
            total_ms = (time.perf_counter_ns() - t_pipeline_start) / 1_000_000.0
            latency.total_ms = round(total_ms, 2)
            latency.target_achieved_50ms = total_ms <= LATENCY_BUDGET_MS
            latency.stretch_achieved_30ms = total_ms <= STRETCH_LATENCY_BUDGET_MS
            latency.target_achieved_200ms = total_ms <= 200.0
            latency.stretch_achieved_150ms = total_ms <= 150.0

            refusal_text = LOCALIZED_REFUSALS.get(lang_code, LOCALIZED_REFUSALS["en"])
            refusal_res = RAGResponse(
                query=cleaned_query,
                detected_language=detected_script,
                answer=refusal_text,
                is_refusal=True,
                grounding=GroundingResult(
                    is_grounded=False,
                    grounding_score=round(align_score if not is_aligned else max_source_score, 4),
                    refusal_triggered=True,
                    refusal_reason=align_reason if not is_aligned else f"Retrieved context relevance ({max_source_score:.2f}) below threshold ({MIN_RETRIEVAL_SCORE:.2f}).",
                ),
                sources=[],
                latency=latency,
                request_id=request_id,
            )

            if ENABLE_RAG_CACHE:
                with self._response_cache_lock:
                    if len(self._response_cache) >= 4096:
                        self._response_cache.popitem(last=False)
                    self._response_cache[cache_key] = refusal_res

            return refusal_res

        # 3. Optional Reranking
        sources, rerank_ms = self.reranker.rerank(cleaned_query, sources, top_k=top_k)
        latency.reranker_ms = round(rerank_ms, 2)

        # 4. Prompt Assembly
        t_prompt_start = time.perf_counter_ns()
        system_prompt, user_msg, resolved_lang = build_rag_prompt(cleaned_query, sources, language_hint=lang_code)
        latency.prompt_construction_ms = round((time.perf_counter_ns() - t_prompt_start) / 1_000_000.0, 2)

        # 5. LLM Generation
        raw_answer, ttft_ms, gen_ms, tok_count, gen_tps, e2e_tps = self.llm_generator.generate_stream(
            cleaned_query, sources, language_hint=resolved_lang
        )
        latency.llm_ttft_ms = round(ttft_ms, 2)
        latency.llm_generation_ms = round(gen_ms, 2)

        # 6. Grounding Check
        grounding_res, ground_ms = self.guardrails.check_grounding(cleaned_query, sources, raw_answer)
        latency.grounding_check_ms = round(ground_ms, 2)

        # 7. Output Sanitization & Refusal Override
        final_answer, _ = self.guardrails.sanitize_output(
            raw_answer,
            is_refusal=grounding_res.refusal_triggered,
            language=resolved_lang,
        )

        # Total latency computation
        total_ms = (time.perf_counter_ns() - t_pipeline_start) / 1_000_000.0
        latency.total_ms = round(total_ms, 2)
        latency.target_achieved_50ms = total_ms <= LATENCY_BUDGET_MS
        latency.stretch_achieved_30ms = total_ms <= STRETCH_LATENCY_BUDGET_MS
        latency.target_achieved_200ms = total_ms <= 200.0
        latency.stretch_achieved_150ms = total_ms <= 150.0

        debug_info = None
        if request.include_debug:
            debug_info = {
                "detected_script": detected_script,
                "candidate_count": len(sources),
                "max_score": sources[0].score if sources else 0.0,
                "system_prompt_len": len(system_prompt),
            }

        response = RAGResponse(
            query=cleaned_query,
            detected_language=detected_script,
            answer=final_answer,
            is_refusal=grounding_res.refusal_triggered,
            grounding=grounding_res,
            sources=sources if not grounding_res.refusal_triggered else [],
            latency=latency,
            request_id=request_id,
            debug_info=debug_info,
        )

        if ENABLE_RAG_CACHE:
            with self._response_cache_lock:
                if len(self._response_cache) >= 4096:
                    self._response_cache.popitem(last=False)
                self._response_cache[cache_key] = response

        return response

    def process_voice_query(self, request: VoiceQueryRequest) -> RAGResponse:
        """
        Execute full Voice-Enabled RAG pipeline: Voice -> STT -> RAG Retrieval -> LLM -> Concurrent TTS.
        """
        t_voice_start = time.perf_counter_ns()

        lang_hint = request.language_hint or request.language
        # 1. STT / Text Query Resolution
        if request.query and request.query.strip():
            transcribed_text = request.query.strip()
            lang_code, _, _ = resolve_query_language(transcribed_text, language_hint=lang_hint)
            detected_lang = lang_code
            stt_ms = 0.0
        elif request.audio_base64:
            transcribed_text, detected_lang, stt_ms = self.stt.transcribe(
                audio_data=request.audio_base64,
                language_hint=lang_hint,
                audio_format=request.audio_format,
            )
            lang_code, _, _ = resolve_query_language(transcribed_text, language_hint=detected_lang)
            detected_lang = lang_code
        else:
            return RAGResponse(
                query="",
                detected_language="Unknown",
                answer="No audio payload or query text provided.",
                is_refusal=True,
                grounding=GroundingResult(is_grounded=False, refusal_triggered=True, refusal_reason="Empty query input"),
                sources=[],
                latency=LatencyBreakdown(total_ms=0.0),
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

        lang_hint = request.language_hint or request.language
        # 1. STT / Text Query Resolution
        if request.query and request.query.strip():
            transcribed_text = request.query.strip()
            lang_code, _, _ = resolve_query_language(transcribed_text, language_hint=lang_hint)
            detected_lang = lang_code
            stt_ms = 0.0
            yield VoiceStreamChunk(event="status", session_id=sid, text="THINKING")
        elif request.audio_base64:
            yield VoiceStreamChunk(event="status", session_id=sid, text="LISTENING")
            transcribed_text, detected_lang, stt_ms = self.stt.transcribe(
                audio_data=request.audio_base64,
                language_hint=lang_hint,
                audio_format=request.audio_format,
            )
            lang_code, _, _ = resolve_query_language(transcribed_text, language_hint=detected_lang)
            detected_lang = lang_code
        else:
            yield VoiceStreamChunk(
                event="error",
                session_id=sid,
                text="No audio payload or query text provided.",
                is_final=True,
            )
            return

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
        sources, ret_latencies = self.hybrid_retriever.search(
            query=cleaned_query, top_k=top_k, target_language=detected_lang
        )
        ret_ms = ret_latencies.get("hybrid_fusion_ms", 15.0)

        # 3. Concurrent LLM + TTS Streaming
        system_prompt, user_msg, resolved_lang = build_rag_prompt(cleaned_query, sources, language_hint=detected_lang)

        def stream_supplier():
            return self.llm_generator.get_stream(system_prompt, user_msg, cleaned_query, sources)

        for event_chunk in self.voice_pipeline.stream_voice_events(
            query=cleaned_query,
            language=resolved_lang or detected_lang or detected_script,
            retrieval_ms=ret_ms,
            llm_stream_generator=stream_supplier,
            session_id=sid,
            sources=sources,
        ):
            if event_chunk.latency:
                event_chunk.latency.stt_ms = round(stt_ms, 2)
            yield event_chunk

    def interrupt(self, session_id: str) -> bool:
        """Interrupt active voice stream for session_id."""
        return self.voice_pipeline.interrupt_session(session_id)
