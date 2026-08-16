"""
evaluation/voice/streaming_buffer.py
------------------------------------
Production-ready Streaming Text Buffer.
Deterministic token accumulation with BPE leading-space boundary detection,
clause/sentence segmentation, and eager Chunk 1 adaptive emission.
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
    """

    def __init__(self, mode: str = "adaptive") -> None:
        """
        Modes:
        - 'tok3_min': 3-token minimum buffering
        - 'adaptive': Eager first phrase at >=3 tokens / clause; clause or sentence thereafter
        """
        self.mode = mode.lower()
        self.accumulated_text = ""
        self.accumulated_tokens = 0
        self.chunks_emitted: list[dict[str, Any]] = []
        self.is_first_chunk = True

    def process_token(self, token: str, token_timestamp_ns: int) -> Optional[dict[str, Any]]:
        if not token:
            return None

        # Check if incoming token marks a word boundary for previously accumulated text
        is_word_boundary = token.startswith((" ", "\t", "\n")) or any(token.startswith(p) for p in ALL_PUNCTUATION)

        chunk_to_emit = None
        if is_word_boundary and self.accumulated_tokens > 0:
            if self._should_emit_at_boundary():
                chunk_to_emit = self.accumulated_text.strip()
                self.accumulated_text = ""
                self.accumulated_tokens = 0

        self.accumulated_text += token
        self.accumulated_tokens += 1

        # Check if accumulated text ends with explicit clause/sentence punctuation
        stripped = self.accumulated_text.rstrip()
        if any(stripped.endswith(term) for term in ALL_PUNCTUATION) and len(stripped) >= 3:
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
        if self.mode == "tok3_min":
            return self.accumulated_tokens >= 3

        if self.mode == "tok5_min":
            return self.accumulated_tokens >= 5

        if self.mode == "sentence":
            stripped = self.accumulated_text.rstrip()
            return any(stripped.endswith(term) for term in SENTENCE_TERMINATORS)

        if self.mode == "adaptive":
            if self.is_first_chunk:
                stripped = self.accumulated_text.rstrip()
                if any(stripped.endswith(term) for term in ALL_PUNCTUATION) and len(stripped) >= 3:
                    return True
                return self.accumulated_tokens >= 3
            else:
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
