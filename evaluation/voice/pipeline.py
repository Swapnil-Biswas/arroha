"""
evaluation/voice/pipeline.py
----------------------------
Real-Time Concurrent Streaming Voice Pipeline.
Producer (LLM Streaming) -> Buffer -> Queue -> Consumer 1 (TTS Worker) -> Audio Queue -> Consumer 2 (Playback Tracker).
Tracks timeline events, inter-chunk latency, starvation gaps, and perceived first-audio response time.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from evaluation.voice.language_router import LanguageRouter
from evaluation.voice.streaming_buffer import StreamingTextBuffer
from evaluation.voice.tts_backend import AudioChunk, TTSBackend

logger = logging.getLogger(__name__)


@dataclass
class VoicePipelineMetrics:
    request_start_ns: int
    retrieval_end_ns: int = 0
    ttft_ns: int = 0
    t3_ns: int = 0
    t5_ns: int = 0
    first_chunk_text_ns: int = 0
    tts_first_audio_ns: int = 0
    llm_end_ns: int = 0
    audio_playback_end_ns: int = 0
    chunks: list[dict[str, Any]] = field(default_factory=list)
    starvation_events: int = 0
    max_queue_depth: int = 0
    audio_continuity_pass: bool = True

    @property
    def user_perceived_latency_ms(self) -> float:
        if self.tts_first_audio_ns and self.request_start_ns:
            return round((self.tts_first_audio_ns - self.request_start_ns) / 1e6, 2)
        return 0.0

    @property
    def ttft_ms(self) -> float:
        if self.ttft_ns and self.request_start_ns:
            return round((self.ttft_ns - self.request_start_ns) / 1e6, 2)
        return 0.0

    @property
    def t3_ms(self) -> float:
        if self.t3_ns and self.request_start_ns:
            return round((self.t3_ns - self.request_start_ns) / 1e6, 2)
        return 0.0

    @property
    def t5_ms(self) -> float:
        if self.t5_ns and self.request_start_ns:
            return round((self.t5_ns - self.request_start_ns) / 1e6, 2)
        return 0.0

    @property
    def total_llm_ms(self) -> float:
        if self.llm_end_ns and self.request_start_ns:
            return round((self.llm_end_ns - self.request_start_ns) / 1e6, 2)
        return 0.0

    @property
    def spoke_before_llm_end(self) -> bool:
        return bool(self.tts_first_audio_ns and self.llm_end_ns and self.tts_first_audio_ns < self.llm_end_ns)


class StreamingVoicePipeline:
    """
    Concurrent Voice Pipeline orchestrating LLM streaming, text chunking, and incremental TTS synthesis.
    """

    def __init__(self, tts_backend: TTSBackend, buffer_mode: str = "adaptive") -> None:
        self.tts_backend = tts_backend
        self.buffer_mode = buffer_mode

    def process_query_stream(
        self,
        query: str,
        language: str,
        retrieval_ms: float,
        llm_stream_generator: Callable[[], Any],
    ) -> VoicePipelineMetrics:
        """
        Executes concurrent streaming pipeline.
        Producer: reads tokens from llm_stream_generator.
        Consumer: synthesizes audio frames as soon as text buffer yields a chunk.
        """
        t_req_ns = time.perf_counter_ns()
        metrics = VoicePipelineMetrics(request_start_ns=t_req_ns)
        metrics.retrieval_end_ns = t_req_ns + int(retrieval_ms * 1e6)

        text_queue: queue.Queue[Optional[dict[str, Any]]] = queue.Queue()
        audio_queue: queue.Queue[Optional[AudioChunk]] = queue.Queue()

        buffer = StreamingTextBuffer(mode=self.buffer_mode)

        # Worker: TTS Consumer
        def tts_worker() -> None:
            c_idx = 1
            while True:
                item = text_queue.get()
                if item is None:
                    audio_queue.put(None)
                    break
                # Synthesize chunk
                t_tts_start = time.perf_counter_ns()
                achunk = self.tts_backend.synthesize_chunk(item["text"], language=language, chunk_index=c_idx)
                t_audio_ready = time.perf_counter_ns()

                if metrics.tts_first_audio_ns == 0:
                    metrics.tts_first_audio_ns = t_audio_ready

                metrics.chunks.append({
                    "chunk_index": c_idx,
                    "text": item["text"],
                    "token_count": item["token_count"],
                    "emitted_at_ms": round((item["timestamp_ns"] - t_req_ns) / 1e6, 2),
                    "audio_ready_at_ms": round((t_audio_ready - t_req_ns) / 1e6, 2),
                    "synthesis_latency_ms": achunk.synthesis_latency_ms,
                    "audio_duration_ms": achunk.audio_duration_ms,
                })
                audio_queue.put(achunk)
                c_idx += 1
                text_queue.task_done()

        tts_thread = threading.Thread(target=tts_worker, daemon=True)
        tts_thread.start()

        # Producer: LLM Token Stream
        tok_count = 0
        try:
            stream = llm_stream_generator()
            for chunk_delta in stream:
                now_ns = time.perf_counter_ns()
                if chunk_delta.choices and len(chunk_delta.choices) > 0:
                    delta = chunk_delta.choices[0].delta
                    if delta and delta.content:
                        tok_count += 1
                        tok_txt = delta.content
                        if metrics.ttft_ns == 0:
                            metrics.ttft_ns = now_ns
                        if tok_count == 3:
                            metrics.t3_ns = now_ns
                        if tok_count == 5:
                            metrics.t5_ns = now_ns

                        emitted = buffer.process_token(tok_txt, now_ns)
                        if emitted:
                            if metrics.first_chunk_text_ns == 0:
                                metrics.first_chunk_text_ns = now_ns
                            text_queue.put(emitted)
        except Exception as e:
            logger.warning("Pipeline LLM stream exception: %s", e)

        t_llm_end_ns = time.perf_counter_ns()
        metrics.llm_end_ns = t_llm_end_ns

        # Flush trailing text
        final_emitted = buffer.flush(t_llm_end_ns)
        if final_emitted:
            if metrics.first_chunk_text_ns == 0:
                metrics.first_chunk_text_ns = t_llm_end_ns
            text_queue.put(final_emitted)

        # Sentinel to terminate TTS worker
        text_queue.put(None)
        tts_thread.join()

        # Audio Playback Continuity Analysis
        playback_head_ms = metrics.user_perceived_latency_ms
        gap_detected = False
        starvations = 0

        for i, c_info in enumerate(metrics.chunks):
            chunk_ready = c_info["audio_ready_at_ms"]
            dur = c_info["audio_duration_ms"]
            if i > 0:
                if chunk_ready > (playback_head_ms + 10.0):
                    gap_detected = True
                    starvations += 1
                    playback_head_ms = chunk_ready + dur
                else:
                    playback_head_ms += dur
            else:
                playback_head_ms += dur

        metrics.audio_playback_end_ns = t_req_ns + int(playback_head_ms * 1e6)
        metrics.audio_continuity_pass = not gap_detected
        metrics.starvation_events = starvations

        return metrics
