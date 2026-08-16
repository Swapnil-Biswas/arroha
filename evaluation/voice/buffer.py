"""
evaluation/voice/buffer.py
--------------------------
Streaming Text Buffer for real-time speech synthesis.
Receives incremental LLM tokens and deterministically emits chunks to TTS
at linguistic and acoustic boundaries without splitting words.
Correctly handles leading-space BPE tokenizers (Qwen / SentencePiece).
"""

from __future__ import annotations

import re
import time
from typing import Any, Optional

SENTENCE_TERMINATORS = (".", "!", "?", "।", "॥", "۔", "\n")
CLAUSE_TERMINATORS = (",", ";", ":", "—", "-", "|", "،", "…")
ALL_PUNCTUATION = SENTENCE_TERMINATORS + CLAUSE_TERMINATORS


class StreamingTextBuffer:
    """
    Deterministic streaming buffer that accumulates tokens and yields speech-ready text chunks.
    Correctly recognizes BPE leading-space word boundaries.
    """

    def __init__(self, strategy: str = "adaptive") -> None:
        self.strategy = strategy.lower()
        self.accumulated_text = ""
        self.accumulated_tokens = 0
        self.chunks_emitted: list[dict[str, Any]] = []
        self.is_first_chunk = True

    def process_token(self, token: str, token_timestamp_ns: int) -> Optional[dict[str, Any]]:
        if not token:
            return None

        # Check if incoming token marks a word boundary for previously accumulated text
        # In BPE tokenizers, space is attached to the START of the incoming token (e.g. " of", " the")
        is_word_boundary = token.startswith((" ", "\t", "\n")) or any(token.startswith(p) for p in ALL_PUNCTUATION)

        # Check if boundary condition is met BEFORE appending incoming token
        chunk_to_emit = None
        if is_word_boundary and self.accumulated_tokens > 0:
            if self._should_emit_at_boundary():
                chunk_to_emit = self.accumulated_text.strip()
                self.accumulated_text = ""
                self.accumulated_tokens = 0

        self.accumulated_text += token
        self.accumulated_tokens += 1

        # Check if text ends with explicit punctuation
        stripped = self.accumulated_text.rstrip()
        if any(stripped.endswith(term) for term in ALL_PUNCTUATION) and len(stripped) >= 3:
            if self.strategy in ("clause", "sentence", "adaptive"):
                if self.strategy == "sentence":
                    if any(stripped.endswith(term) for term in SENTENCE_TERMINATORS):
                        chunk_to_emit = self.accumulated_text.strip()
                        self.accumulated_text = ""
                        self.accumulated_tokens = 0
                else:
                    chunk_to_emit = self.accumulated_text.strip()
                    self.accumulated_text = ""
                    self.accumulated_tokens = 0

        if chunk_to_emit:
            chunk_record = {
                "chunk_index": len(self.chunks_emitted) + 1,
                "text": chunk_to_emit,
                "token_count": self.accumulated_tokens,
                "timestamp_ns": token_timestamp_ns,
                "char_length": len(chunk_to_emit),
                "is_first_chunk": self.is_first_chunk,
            }
            self.chunks_emitted.append(chunk_record)
            self.is_first_chunk = False
            return chunk_record

        return None

    def _should_emit_at_boundary(self) -> bool:
        if self.strategy == "sentence":
            stripped = self.accumulated_text.rstrip()
            return any(stripped.endswith(term) for term in SENTENCE_TERMINATORS)

        if self.strategy == "clause":
            stripped = self.accumulated_text.rstrip()
            return any(stripped.endswith(term) for term in ALL_PUNCTUATION) and len(stripped) >= 3

        if self.strategy == "tok3_min":
            return self.accumulated_tokens >= 3

        if self.strategy == "tok5_min":
            return self.accumulated_tokens >= 5

        if self.strategy == "adaptive":
            if self.is_first_chunk:
                # Eager first chunk: 3 tokens or any clause
                stripped = self.accumulated_text.rstrip()
                if any(stripped.endswith(term) for term in ALL_PUNCTUATION) and len(stripped) >= 3:
                    return True
                return self.accumulated_tokens >= 3
            else:
                # Subsequent chunks: 5 tokens or clause
                stripped = self.accumulated_text.rstrip()
                if any(stripped.endswith(term) for term in ALL_PUNCTUATION) and len(stripped) >= 4:
                    return True
                return self.accumulated_tokens >= 5

        return False

    def flush(self, timestamp_ns: Optional[int] = None) -> Optional[dict[str, Any]]:
        remaining = self.accumulated_text.strip()
        if not remaining:
            return None

        ts = timestamp_ns or time.perf_counter_ns()
        chunk_record = {
            "chunk_index": len(self.chunks_emitted) + 1,
            "text": remaining,
            "token_count": self.accumulated_tokens,
            "timestamp_ns": ts,
            "char_length": len(remaining),
            "is_first_chunk": self.is_first_chunk,
            "is_final_flush": True,
        }
        self.chunks_emitted.append(chunk_record)
        self.accumulated_text = ""
        self.accumulated_tokens = 0
        self.is_first_chunk = False
        return chunk_record
