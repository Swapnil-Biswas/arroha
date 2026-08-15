"""
app/guardrails/input.py
-----------------------
Input query guardrails:
- Sanitization & empty query detection
- Query length boundaries
- Injection / adversarial pattern filtering
- Script & language identification
"""

from __future__ import annotations

import re
import time
from typing import Optional

from ingestion.preprocess import clean_multilingual_text, detect_script

# Common prompt injection / system override patterns
INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+a\s+different\s+model", re.IGNORECASE),
    re.compile(r"system\s*:\s*override", re.IGNORECASE),
    re.compile(r"<\|im_start\|>", re.IGNORECASE),
    re.compile(r"<\|im_end\|>", re.IGNORECASE),
]


class InputGuardrail:
    """
    Validates and sanitizes incoming user queries before retrieval.
    """

    def validate(
        self,
        query: str,
        language_hint: Optional[str] = None,
    ) -> tuple[bool, str, str, Optional[str], float]:
        """
        Validate input query.
        Returns (is_valid, cleaned_query, detected_script, error_reason, latency_ms).
        """
        t0 = time.perf_counter_ns()

        if not query or not query.strip():
            latency_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            return False, "", "Unknown", "Query cannot be empty or whitespace only.", latency_ms

        cleaned = clean_multilingual_text(query)
        if len(cleaned) < 2:
            latency_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            return False, cleaned, "Unknown", "Query is too short to process.", latency_ms

        if len(cleaned) > 1000:
            cleaned = cleaned[:1000]

        # Check prompt injection patterns
        for pattern in INJECTION_PATTERNS:
            if pattern.search(cleaned):
                latency_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
                return False, cleaned, "Unknown", "Adversarial or system override pattern detected.", latency_ms

        script = detect_script(cleaned)
        latency_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        return True, cleaned, script, None, latency_ms
