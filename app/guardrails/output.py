"""
app/guardrails/output.py
------------------------
Output guardrails for sanitizing, formatting, and bounding generated answers.
"""

from __future__ import annotations

import time
from typing import Optional

from app.guardrails.grounding import LOCALIZED_REFUSALS
from ingestion.preprocess import clean_multilingual_text


class OutputGuardrail:
    """
    Validates, bounds, and cleans LLM-generated responses with strict refusal overrides.
    """

    def __init__(self, max_length: int = 500) -> None:
        self.max_length = max_length

    def validate_and_clean(
        self,
        raw_answer: str,
        is_refusal: bool = False,
        language: str = "en",
    ) -> tuple[str, float]:
        """
        Clean and bound the generated answer, ensuring clean localized refusal if triggered.
        Returns (cleaned_answer, latency_ms).
        """
        t0 = time.perf_counter_ns()

        if is_refusal:
            refusal_text = LOCALIZED_REFUSALS.get(language, LOCALIZED_REFUSALS["en"])
            latency_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            return refusal_text, latency_ms

        cleaned = clean_multilingual_text(raw_answer)

        # Strip any leaked prompt headers or template markers
        for prefix in [
            "Factual Answer (in",
            "Factual Answer (strictly in",
            "Factual Answer:",
            "Answer:",
            "CRITICAL RULES:",
            "The critical rules are:",
            "User Question:",
            "Retrieved Context:",
            "[INSUFFICIENT_CONTEXT]",
            "insufficient_context",
        ]:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].lstrip(" :)[]-–")

        if not cleaned or "insufficient" in cleaned.lower() or "enough information" in cleaned.lower():
            refusal_text = LOCALIZED_REFUSALS.get(language, LOCALIZED_REFUSALS["en"])
            latency_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            return refusal_text, latency_ms

        if len(cleaned) > self.max_length:
            cleaned = cleaned[:self.max_length].rstrip() + "..."

        latency_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        return cleaned, latency_ms
