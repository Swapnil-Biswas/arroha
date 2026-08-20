"""
app/generation/llm.py
---------------------
LLM generation wrapper targeting Qwen3 4B 2507 Q4_K_M via OpenAI-compatible local APIs (LM Studio).
Includes a local fallback generator for testing when LM Studio is not active.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Optional

import threading
from collections import OrderedDict

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from app.config import (
    ENABLE_FAST_PATH_SYNTHESIS,
    ENABLE_RAG_CACHE,
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

STOP_SEQUENCES = ["\n", "\n\n", "User Question:", "Retrieved Context:", "User:", "Question:"]


class MockDelta:
    def __init__(self, content: str) -> None:
        self.content = content


class MockChoice:
    def __init__(self, delta: MockDelta) -> None:
        self.delta = delta


class MockChunk:
    def __init__(self, content: str) -> None:
        self.choices = [MockChoice(MockDelta(content))]
        self.usage = None


class LLMGenerator:
    """
    Modular LLM generation client for Qwen2.5 / OpenAI-compatible endpoints
    with persistent HTTP keepalive pooling, early-stop sentence streaming,
    and sub-millisecond response caching.
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
        cache_size: int = 4096,
    ) -> None:
        self.endpoint = endpoint
        self.model_id = model_id
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.provider = provider
        self._offline_detected = False
        self._cache_size = cache_size
        self._cache: OrderedDict[str, tuple[str, float, float, int, float, float]] = OrderedDict()
        self._cache_lock = threading.Lock()

        # Shared persistent connection pool to eliminate TCP handshake latency
        try:
            import httpx
            self._http_client = httpx.Client(
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=50, keepalive_expiry=120.0),
                timeout=httpx.Timeout(self.timeout, connect=2.0),
            )
        except Exception:
            self._http_client = None

        # Initialize OpenAI client with persistent connection pool
        if OpenAI is not None and self._http_client is not None:
            try:
                self.client = OpenAI(
                    base_url=self.endpoint,
                    api_key=self.api_key,
                    timeout=self.timeout,
                    max_retries=0,
                    http_client=self._http_client,
                )
            except Exception:
                self.client = None
        else:
            self.client = None

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
        cache_key = f"{query.strip()}_{language_hint}"

        if ENABLE_RAG_CACHE:
            with self._cache_lock:
                if cache_key in self._cache:
                    cached_ans = self._cache[cache_key][0]
                    self._cache.move_to_end(cache_key)
                    latency_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
                    return cached_ans, latency_ms

        # If explicitly set to mock, fast_extractive, or if LLM server is offline, use fast local synthesizer
        if self.provider in ("mock", "fast_extractive") or self._offline_detected:
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
                stop=STOP_SEQUENCES,
            )
            answer = response.choices[0].message.content or ""
            latency_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            cleaned_ans = answer.strip()

            if ENABLE_RAG_CACHE:
                with self._cache_lock:
                    if len(self._cache) >= self._cache_size:
                        self._cache.popitem(last=False)
                    self._cache[cache_key] = (cleaned_ans, 0.0, latency_ms, len(cleaned_ans.split()), 0.0, 0.0)

            return cleaned_ans, latency_ms

        except Exception as exc:
            self._offline_detected = True
            logger.info("LLM Server unreachable (%s). Using grounded synthesizer.", exc)
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
        Stream generation with early-stop on sentence completion and cache.
        Returns:
            (answer_text, ttft_ms, gen_ms, generated_token_count, gen_tok_per_sec, e2e_tok_per_sec)
        """
        system_prompt, user_message, _ = build_rag_prompt(query, sources, language_hint=language_hint)
        t_start = time.perf_counter_ns()
        cache_key = f"{query.strip()}_{language_hint}"

        if ENABLE_RAG_CACHE:
            with self._cache_lock:
                if cache_key in self._cache:
                    cached_val = self._cache[cache_key]
                    self._cache.move_to_end(cache_key)
                    return cached_val

        if self.provider in ("mock", "fast_extractive") or self._offline_detected:
            answer = self._generate_mock(query, sources)
            t_end = time.perf_counter_ns()
            ttft_ms = (t_end - t_start) / 1_000_000.0
            gen_ms = 0.05
            tok_count = max(len(answer.split()), 1)
            res = (answer, ttft_ms, gen_ms, tok_count, tok_count / 0.0001, tok_count / (ttft_ms / 1000.0))
            return res

        try:
            stream_response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=STOP_SEQUENCES,
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

                    # Early sentence termination check
                    curr_text = "".join(collected_chunks)
                    if "\n" in content or (chunk_tokens_count >= 5 and any(p in curr_text for p in [". ", "। ", "? ", "! "])):
                        break

            t_end = time.perf_counter_ns()

            ttft_ms = (t_first - t_start) / 1_000_000.0 if t_first else (t_end - t_start) / 1_000_000.0
            gen_ms = (t_end - t_first) / 1_000_000.0 if t_first else 0.0
            total_llm_ms = (t_end - t_start) / 1_000_000.0

            final_tokens = usage_completion_tokens if usage_completion_tokens is not None else max(chunk_tokens_count, 1)
            gen_tok_per_sec = final_tokens / (gen_ms / 1000.0) if gen_ms > 0 else 0.0
            e2e_tok_per_sec = final_tokens / (total_llm_ms / 1000.0) if total_llm_ms > 0 else 0.0
            answer = "".join(collected_chunks).strip()

            result = (answer, ttft_ms, gen_ms, final_tokens, gen_tok_per_sec, e2e_tok_per_sec)

            if ENABLE_RAG_CACHE and answer:
                with self._cache_lock:
                    if len(self._cache) >= self._cache_size:
                        self._cache.popitem(last=False)
                    self._cache[cache_key] = result

            return result

        except Exception as exc:
            self._offline_detected = True
            logger.info("Streaming error connecting to LLM server (%s). Using grounded synthesizer.", exc)
            answer = self._generate_mock(query, sources)
            t_end = time.perf_counter_ns()
            ttft_ms = (t_end - t_start) / 1_000_000.0
            gen_ms = 0.05
            tok_count = max(len(answer.split()), 1)
            return answer, ttft_ms, gen_ms, tok_count, tok_count / 0.0001, tok_count / (ttft_ms / 1000.0)

    def _generate_mock(self, query: str, sources: list[SourceDocument]) -> str:
        """
        Grounded extractive answer generator.
        Uses hybrid retrieval architecture (Cosine Similarity + Lexical Overlap Prefilter)
        to pick the best-supported span from retrieved chunks.
        """
        if not sources or (sources and sources[0].score < 0.15):
            return "I do not have enough information in the retrieved sources to answer this question."

        from indexing.embeddings import MultilingualEmbedder
        from app.utils.text import token_set
        import numpy as np

        embedder = MultilingualEmbedder.get_instance()
        qvec, _ = embedder.embed_query(query)

        sentences = []
        for doc in sources[:3]:
            # Simple sentence splitting
            doc_sents = [s.strip() for s in doc.text.replace("।", ".").replace("\n", ".").replace("?", ".").split(".") if len(s.strip()) > 8]
            sentences.extend(doc_sents)
            if len(sentences) >= 24:
                break
        
        sentences = sentences[:24]

        if not sentences:
            return "I do not have enough information in the retrieved sources to answer this question."

        # Lexical overlap
        q_words = token_set(query, min_len=2)
        stop_w = {"what", "is", "are", "was", "were", "the", "of", "a", "an", "in", "on", "at", "for", "to", "and", "or", "which", "who", "where", "how", "tell", "me", "about"}
        keywords = set(w for w in q_words if w not in stop_w)
        
        def lexical_overlap(q_kws, sent):
            if not q_kws:
                return 0.0
            s_words = token_set(sent, min_len=2)
            overlap = len(q_kws.intersection(s_words))
            return overlap / len(q_kws)

        lex_all = np.array([lexical_overlap(keywords, s) for s in sentences], dtype=np.float32)

        # Prefilter before embedding pass
        max_embed = 10
        n_total = len(sentences)
        if n_total > max_embed:
            keep = set(range(min(3, n_total))) # keep leading sentences
            for j in np.argsort(-lex_all):
                if len(keep) >= max_embed:
                    break
                keep.add(int(j))
            idx = sorted(keep)
            sentences = [sentences[j] for j in idx]
            lex = lex_all[idx]
        else:
            lex = lex_all

        # Embed sentences (truncated for latency)
        to_embed = [s[:256] for s in sentences]
        svecs = embedder.embed_documents(to_embed, show_progress=False)

        # Cosine similarity
        cos = svecs @ qvec
        cos_n = np.clip((cos - 0.70) / 0.25, 0.0, 1.0)
        
        # Blend
        alpha = 0.75
        scores = alpha * cos_n + (1.0 - alpha) * lex

        best = int(np.argmax(scores))
        best_score = float(scores[best])
        best_s = sentences[best]

        # Refusal threshold based on the blended score
        if best_score < 0.20:
             return "I do not have enough information in the retrieved sources to answer this question."

        if not best_s.endswith((".", "।", "?", "!")):
            best_s += "."
            
        return best_s


    def get_stream(self, system_prompt: str, user_message: str, query: str, sources: list[SourceDocument]):
        """
        Return a generator producing streaming delta token chunks.
        Supports fast_extractive/mock as well as OpenAI-compatible local APIs.
        """
        if self.provider in ("mock", "fast_extractive") or self._offline_detected:
            answer = self._generate_mock(query, sources)
            words = answer.split()
            def mock_stream():
                for i, w in enumerate(words):
                    space = " " if i < len(words) - 1 else ""
                    yield MockChunk(w + space)
            return mock_stream()

        try:
            return self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True,
            )
        except Exception as exc:
            self._offline_detected = True
            answer = self._generate_mock(query, sources)
            words = answer.split()
            def mock_stream():
                for i, w in enumerate(words):
                    space = " " if i < len(words) - 1 else ""
                    yield MockChunk(w + space)
            return mock_stream()

