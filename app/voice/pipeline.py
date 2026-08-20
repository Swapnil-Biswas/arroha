"""
app/voice/pipeline.py
---------------------
Concurrent Real-Time Streaming Voice Pipeline for ARROHA.
Coordinates:
- Producer: LLM Token Stream
- Incremental Token Buffer (BPE Word Boundaries)
- Consumer 1: Concurrent TTS Synthesis Queue
- Consumer 2: Audio Chunk Queue
- Interruption Handle: Instant cancellation & queue draining
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, Iterator, Optional

from app.schemas.response import LatencyBreakdown, SourceDocument, VoiceStreamChunk
from app.voice.language_router import LanguageRouter
from app.voice.streaming_buffer import StreamingTextBuffer
from app.voice.tts_backend import AudioChunk, TTSBackend

logger = logging.getLogger(__name__)


@dataclass
class SessionState:
    session_id: str
    is_active: bool = True
    is_interrupted: bool = False
    created_at: float = field(default_factory=time.time)


class StreamingVoicePipeline:
    """
    Concurrent Voice Pipeline orchestrating LLM streaming, text chunking, and incremental TTS synthesis.
    """

    def __init__(self, tts_backend: TTSBackend, buffer_mode: str = "adaptive") -> None:
        self.tts_backend = tts_backend
        self.buffer_mode = buffer_mode
        self._active_sessions: dict[str, SessionState] = {}
        self._lock = threading.Lock()

    def register_session(self, session_id: str) -> SessionState:
        with self._lock:
            state = SessionState(session_id=session_id)
            self._active_sessions[session_id] = state
            return state

    def interrupt_session(self, session_id: str) -> bool:
        """Interrupt active speech/generation for a session."""
        with self._lock:
            if session_id in self._active_sessions:
                self._active_sessions[session_id].is_interrupted = True
                self._active_sessions[session_id].is_active = False
                logger.info("Session %s interrupted (barge-in triggered).", session_id)
                return True
        return False

    def stream_voice_events(
        self,
        query: str,
        language: str,
        retrieval_ms: float,
        llm_stream_generator: Callable[[], Any],
        session_id: str = "default",
        sources: Optional[list[SourceDocument]] = None,
    ) -> Generator[VoiceStreamChunk, None, None]:
        """
        Yields real-time SSE event stream chunks:
        1. status: THINKING
        2. token: delta text tokens as generated
        3. audio_chunk: synthesized speech frames as ready
        4. metrics / done
        """
        t_req_ns = time.perf_counter_ns()
        session_state = self.register_session(session_id)

        # Notify THINKING state
        yield VoiceStreamChunk(
            event="status",
            session_id=session_id,
            text="THINKING",
            language=language,
        )

        text_queue: queue.Queue[Optional[dict[str, Any]]] = queue.Queue()
        audio_queue: queue.Queue[Optional[AudioChunk]] = queue.Queue()
        buffer = StreamingTextBuffer(mode=self.buffer_mode)

        # Worker: TTS Consumer
        def tts_worker() -> None:
            c_idx = 1
            while True:
                if session_state.is_interrupted:
                    audio_queue.put(None)
                    break
                item = text_queue.get()
                if item is None:
                    audio_queue.put(None)
                    break
                if session_state.is_interrupted:
                    text_queue.task_done()
                    audio_queue.put(None)
                    break

                achunk = self.tts_backend.synthesize_chunk(item["text"], language=language, chunk_index=c_idx)
                audio_queue.put(achunk)
                c_idx += 1
                text_queue.task_done()

        tts_thread = threading.Thread(target=tts_worker, daemon=True)
        tts_thread.start()

        # Producer & Stream Dispatcher
        t_first_audio_ns = None
        t1_ns = None
        full_tokens: list[str] = []
        tok_count = 0

        try:
            stream = llm_stream_generator()
            for chunk_delta in stream:
                if session_state.is_interrupted:
                    yield VoiceStreamChunk(event="status", session_id=session_id, text="INTERRUPTED")
                    break

                now_ns = time.perf_counter_ns()
                if chunk_delta.choices and len(chunk_delta.choices) > 0:
                    delta = chunk_delta.choices[0].delta
                    if delta and delta.content:
                        tok_count += 1
                        tok_txt = delta.content
                        full_tokens.append(tok_txt)
                        if t1_ns is None:
                            t1_ns = now_ns

                        # Emit token delta to client
                        yield VoiceStreamChunk(
                            event="token",
                            session_id=session_id,
                            delta=tok_txt,
                            chunk_index=tok_count,
                        )

                        emitted = buffer.process_token(tok_txt, now_ns)
                        if emitted:
                            text_queue.put(emitted)

                # Check if TTS has produced any audio chunks while LLM generated
                while not audio_queue.empty():
                    achunk = audio_queue.get_nowait()
                    if achunk is not None:
                        if t_first_audio_ns is None:
                            t_first_audio_ns = achunk.created_at_ns if (achunk.created_at_ns and achunk.created_at_ns > 0) else time.perf_counter_ns()
                            yield VoiceStreamChunk(event="status", session_id=session_id, text="SPEAKING")

                        yield VoiceStreamChunk(
                            event="audio_chunk",
                            session_id=session_id,
                            text=achunk.text,
                            audio_base64=achunk.audio_base64,
                            chunk_index=achunk.chunk_index,
                            audio_duration_ms=achunk.audio_duration_ms,
                            synthesis_latency_ms=achunk.synthesis_latency_ms,
                        )
                        audio_queue.task_done()
        except Exception as exc:
            logger.warning("Voice stream generator exception: %s", exc)

        t_llm_end_ns = time.perf_counter_ns()

        # Flush remaining text buffer
        final_emitted = buffer.flush(t_llm_end_ns)
        if final_emitted and not session_state.is_interrupted:
            text_queue.put(final_emitted)

        # Sentinel to finish TTS worker
        text_queue.put(None)
        tts_thread.join()

        # Drain remaining audio chunks
        while True:
            achunk = audio_queue.get()
            if achunk is None:
                break
            if t_first_audio_ns is None:
                t_first_audio_ns = achunk.created_at_ns if (achunk.created_at_ns and achunk.created_at_ns > 0) else time.perf_counter_ns()
                yield VoiceStreamChunk(event="status", session_id=session_id, text="SPEAKING")

            yield VoiceStreamChunk(
                event="audio_chunk",
                session_id=session_id,
                text=achunk.text,
                audio_base64=achunk.audio_base64,
                chunk_index=achunk.chunk_index,
                audio_duration_ms=achunk.audio_duration_ms,
                synthesis_latency_ms=achunk.synthesis_latency_ms,
            )
            audio_queue.task_done()

        t_end_ns = time.perf_counter_ns()
        if t_first_audio_ns is None:
            t_first_audio_ns = t_end_ns

        # Calculate final latency breakdown
        first_audio_lat_ms = round((t_first_audio_ns - t_req_ns) / 1e6, 2)
        ttft_ms = round(((t1_ns or t_llm_end_ns) - t_req_ns) / 1e6, 2)
        total_ms = round((t_end_ns - t_req_ns) / 1e6, 2)

        lat = LatencyBreakdown(
            stt_ms=0.0,
            vector_retrieval_ms=retrieval_ms,
            llm_ttft_ms=ttft_ms,
            first_audio_latency_ms=first_audio_lat_ms,
            total_ms=total_ms,
            target_achieved_200ms=first_audio_lat_ms <= 200.0,
            stretch_achieved_150ms=first_audio_lat_ms <= 150.0,
        )

        yield VoiceStreamChunk(
            event="done",
            session_id=session_id,
            text="".join(full_tokens).strip(),
            language=language,
            sources=sources,
            latency=lat,
            is_final=True,
        )
