"""
app/guardrails/grounding.py
---------------------------
Grounding and hallucination detection.
Verifies that generated answers are strictly supported by retrieved context snippets.
"""

from __future__ import annotations

import re
import time
from typing import Optional

from app.config import GROUNDING_SIMILARITY_THRESHOLD, MIN_RETRIEVAL_SCORE
from app.schemas.response import GroundingResult, SourceDocument

REFUSAL_PHRASES = [
    "do not have enough information",
    "does not contain enough information",
    "पर्याप्त जानकारी नहीं है", # Hindi
    "পর্যাপ্ত তথ্য নেই",         # Bengali
    "போதுமான தகவல் இல்லை",       # Tamil
    "సరిపోవు సమాచారం లేదు",     # Telugu
    "पुरेशी माहिती नाही",        # Marathi
    "પૂરતી માહિતી નથી",          # Gujarati
    "cannot answer",
    "insufficient evidence",
]


class GroundingChecker:
    """
    Evaluates factual grounding and hallucination risk in generated answers.
    """

    def __init__(
        self,
        min_retrieval_score: float = MIN_RETRIEVAL_SCORE,
        similarity_threshold: float = GROUNDING_SIMILARITY_THRESHOLD,
    ) -> None:
        self.min_retrieval_score = min_retrieval_score
        self.similarity_threshold = similarity_threshold

    def check(
        self,
        query: str,
        sources: list[SourceDocument],
        generated_answer: str,
    ) -> tuple[GroundingResult, float]:
        """
        Verify grounding of generated answer against sources.
        Returns (GroundingResult, latency_ms).
        """
        t0 = time.perf_counter_ns()

        # 1. Check for explicit refusal in generated text
        answer_lower = generated_answer.lower()
        for phrase in REFUSAL_PHRASES:
            if phrase.lower() in answer_lower:
                latency_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
                return (
                    GroundingResult(
                        is_grounded=True,
                        grounding_score=1.0,
                        refusal_triggered=True,
                        refusal_reason="Model explicitly recognized insufficient context and refused.",
                    ),
                    latency_ms,
                )

        # 2. Check if context was too weak / missing
        if not sources or max((s.score for s in sources), default=0.0) < self.min_retrieval_score:
            latency_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            return (
                GroundingResult(
                    is_grounded=False,
                    grounding_score=0.1,
                    refusal_triggered=True,
                    refusal_reason="Retrieved context has insufficient relevance score.",
                ),
                latency_ms,
            )

        # 3. Lexical overlap check between generated answer and concatenated context
        combined_context = " ".join([s.text for s in sources]).lower()
        answer_words = [w for w in re.findall(r"[\w\u0900-\u0D7F]+", answer_lower) if len(w) > 2]

        if not answer_words:
            overlap_ratio = 1.0
        else:
            matched_words = [w for w in answer_words if w in combined_context]
            overlap_ratio = len(matched_words) / len(answer_words)

        is_grounded = overlap_ratio >= self.similarity_threshold

        result = GroundingResult(
            is_grounded=is_grounded,
            grounding_score=round(overlap_ratio, 4),
            refusal_triggered=not is_grounded,
            refusal_reason=None if is_grounded else f"Lexical grounding overlap ({overlap_ratio:.2f}) below threshold ({self.similarity_threshold}).",
        )

        latency_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        return result, latency_ms
