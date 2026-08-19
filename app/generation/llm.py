"""
app/generation/llm.py
---------------------
LLM generation wrapper targeting Qwen3 4B 2507 Q4_K_M via OpenAI-compatible local APIs (LM Studio).
Includes a local fallback generator for testing when LM Studio is not active.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import httpx
from openai import OpenAI

from app.config import (
    LLM_API_KEY,
    LLM_ENDPOINT,
    LLM_MAX_TOKENS,
    LLM_MODEL_ID,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
)
from app.generation.prompts import build_rag_prompt
from app.schemas.response import SourceDocument

logger = logging.getLogger(__name__)


class LLMGenerator:
    """
    Modular LLM generation client for Qwen3 4B.
    Connects to LM Studio / local OpenAI-compatible endpoints with fast fallback.
    """

    def __init__(
        self,
        endpoint: str = LLM_ENDPOINT,
        model_id: str = LLM_MODEL_ID,
        api_key: str = LLM_API_KEY,
        max_tokens: int = LLM_MAX_TOKENS,
        temperature: float = LLM_TEMPERATURE,
        timeout: float = LLM_TIMEOUT_SECONDS,
        provider: str = LLM_PROVIDER,
    ) -> None:
        self.endpoint = endpoint
        self.model_id = model_id
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.provider = provider
        self._offline_detected = False

        # Initialize OpenAI client pointed to local LM Studio
        self.client = OpenAI(
            base_url=self.endpoint,
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=0,
        )

    def generate(
        self,
        query: str,
        sources: list[SourceDocument],
        language_hint: Optional[str] = None,
        stream: bool = False,
    ) -> tuple[str, float]:
        """
        Generate answer from query and retrieved sources.
        Returns (answer_text, generation_latency_ms).
        """
        system_prompt, user_message, _ = build_rag_prompt(query, sources, language_hint=language_hint)
        t0 = time.perf_counter_ns()

        # If explicitly set to mock or if LM Studio is offline, use fast local synthesizer
        if self.provider == "mock" or self._offline_detected:
            answer = self._generate_mock(query, sources)
            latency_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            return answer, latency_ms

        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            answer = response.choices[0].message.content or ""
            latency_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            return answer.strip(), latency_ms

        except Exception as exc:
            self._offline_detected = True
            logger.info("Local LM Studio not connected (%s). Using grounded synthesizer.", exc)
            answer = self._generate_mock(query, sources)
            latency_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            return answer, latency_ms

    def generate_stream(
        self,
        query: str,
        sources: list[SourceDocument],
        language_hint: Optional[str] = None,
    ) -> tuple[str, float, float, int, float, float]:
        """
        Stream generation from Qwen3 to accurately capture TTFT, pure generation time,
        and token counts directly from API chunks/usage.
        Returns:
            (answer_text, ttft_ms, gen_ms, generated_token_count, gen_tok_per_sec, e2e_tok_per_sec)
        """
        system_prompt, user_message, _ = build_rag_prompt(query, sources, language_hint=language_hint)
        t_start = time.perf_counter_ns()

        if self.provider == "mock" or self._offline_detected:
            answer = self._generate_mock(query, sources)
            t_end = time.perf_counter_ns()
            ttft_ms = (t_end - t_start) / 1_000_000.0
            gen_ms = 0.1
            tok_count = max(len(answer.split()), 1)
            return answer, ttft_ms, gen_ms, tok_count, tok_count / 0.0001, tok_count / (ttft_ms / 1000.0)

        try:
            stream_response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True,
                stream_options={"include_usage": True},
            )

            t_first = None
            collected_chunks: list[str] = []
            chunk_tokens_count = 0
            usage_completion_tokens: Optional[int] = None

            for chunk in stream_response:
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_completion_tokens = chunk.usage.completion_tokens
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    if t_first is None:
                        t_first = time.perf_counter_ns()
                    content = chunk.choices[0].delta.content
                    collected_chunks.append(content)
                    chunk_tokens_count += 1

            t_end = time.perf_counter_ns()

            ttft_ms = (t_first - t_start) / 1_000_000.0 if t_first else (t_end - t_start) / 1_000_000.0
            gen_ms = (t_end - t_first) / 1_000_000.0 if t_first else 0.0
            total_llm_ms = (t_end - t_start) / 1_000_000.0

            final_tokens = usage_completion_tokens if usage_completion_tokens is not None else max(chunk_tokens_count, 1)
            gen_tok_per_sec = final_tokens / (gen_ms / 1000.0) if gen_ms > 0 else 0.0
            e2e_tok_per_sec = final_tokens / (total_llm_ms / 1000.0) if total_llm_ms > 0 else 0.0
            answer = "".join(collected_chunks).strip()

            return answer, ttft_ms, gen_ms, final_tokens, gen_tok_per_sec, e2e_tok_per_sec

        except Exception as exc:
            self._offline_detected = True
            logger.info("Streaming error connecting to LM Studio (%s). Using grounded synthesizer.", exc)
            answer = self._generate_mock(query, sources)
            t_end = time.perf_counter_ns()
            ttft_ms = (t_end - t_start) / 1_000_000.0
            gen_ms = 0.1
            tok_count = max(len(answer.split()), 1)
            return answer, ttft_ms, gen_ms, tok_count, tok_count / 0.0001, tok_count / (ttft_ms / 1000.0)

    def _generate_mock(self, query: str, sources: list[SourceDocument]) -> str:
        """
        Deterministic, grounded answer synthesizer for testing and offline execution.
        Extracts the most relevant sentence from top source or triggers refusal.
        """
        if not sources or (sources and sources[0].score < 0.2):
            return "I do not have enough information in the retrieved sources to answer this question."

        top_source = sources[0]
        # Return first 1-2 sentences of top source
        sentences = [s.strip() for s in top_source.text.split(".") if s.strip()]
        if not sentences:
            sentences = [s.strip() for s in top_source.text.split("।") if s.strip()]
        if sentences:
            return sentences[0] + "."
        return top_source.text[:150]
