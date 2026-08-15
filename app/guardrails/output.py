"""
app/guardrails/output.py
------------------------
Output guardrails for sanitizing, formatting, and bounding generated answers.
"""

from __future__ import annotations

import time
from typing import Optional

from ingestion.preprocess import clean_multilingual_text


class OutputGuardrail:
    """
    Validates and formats LLM generated responses.
    """

    def __init__(self, max_length: int = 500) -> None:
        self.max_length = max_length

    def validate_and_clean(
        self,
        raw_answer: str,
        is_refusal: bool = False,
    ) -> tuple[str, float]:
        """
        Clean and bound the generated answer.
        Returns (cleaned_answer, latency_ms).
        """
        t0 = time.perf_counter_ns()
        cleaned = clean_multilingual_text(raw_answer)

        if is_refusal:
            if not cleaned:
                cleaned = "I do not have enough information in the retrieved sources to answer this question."
            latency_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            return cleaned, latency_ms

        if len(cleaned) > self.max_length:
            cleaned = cleaned[:self.max_length].rstrip() + "..."

        latency_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        return cleaned, latency_ms
