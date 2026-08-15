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
)
from app.voice.stt import SpeechToTextEngine

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Core RAG Pipeline coordinating all components.
    """

    def __init__(
        self,
        hybrid_retriever: Optional[HybridRetriever] = None,
        llm_generator: Optional[LLMGenerator] = None,
        guardrails_validator: Optional[GuardrailsValidator] = None,
        stt_engine: Optional[SpeechToTextEngine] = None,
        reranker: Optional[Reranker] = None,
    ) -> None:
        self.hybrid_retriever = hybrid_retriever or HybridRetriever()
        self.llm_generator = llm_generator or LLMGenerator()
        self.guardrails = guardrails_validator or GuardrailsValidator()
        self.stt = stt_engine or SpeechToTextEngine()
        self.reranker = reranker or Reranker()

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
        Execute full Voice-Enabled RAG pipeline: Voice -> STT -> Text RAG.
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

        # Update STT latency and overall total
        response.latency.stt_ms = round(stt_ms, 2)
        total_voice_ms = (time.perf_counter_ns() - t_voice_start) / 1_000_000.0
        response.latency.total_ms = round(total_voice_ms, 2)
        response.latency.target_achieved_200ms = total_voice_ms <= LATENCY_BUDGET_MS
        response.latency.stretch_achieved_150ms = total_voice_ms <= STRETCH_LATENCY_BUDGET_MS

        return response
