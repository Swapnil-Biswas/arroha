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
    re.compile(r"ignore\s+(all\s+)?(previous\s+|prior\s+)?instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a\s+)?(different\s+model|DAN|unrestricted)", re.IGNORECASE),
    re.compile(r"system\s*:\s*override", re.IGNORECASE),
    re.compile(r"<\|im_start\|>", re.IGNORECASE),
    re.compile(r"<\|im_end\|>", re.IGNORECASE),
    re.compile(r"\[/?INST\]", re.IGNORECASE),
    re.compile(r"developer\s+mode\s+(enabled|on|activate)", re.IGNORECASE),
    re.compile(r"bypass\s+(all\s+)?(safety\s+|content\s+)?(filters|rules|guardrails)", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous\s+)?(rules|constraints|instructions)", re.IGNORECASE),
    re.compile(r"(reveal|print|show|leak)\s+(the\s+|your\s+)?(system\s+prompt|hidden\s+instructions)", re.IGNORECASE),
    re.compile(r"pretend\s+(you\s+have\s+no|there\s+are\s+no)\s+(rules|guidelines|restrictions)", re.IGNORECASE),
    re.compile(r"do\s+anything\s+now", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
]


class InputGuardrail:
    """
    Validates and sanitizes incoming user queries before retrieval.
    Enforces injection defense, boundary checks, and script detection.
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

        # Check prompt injection and adversarial jailbreak patterns
        for pattern in INJECTION_PATTERNS:
            if pattern.search(cleaned):
                latency_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
                return False, cleaned, "Unknown", "Adversarial or system override pattern detected. Request cannot be processed.", latency_ms

        script = detect_script(cleaned)
        latency_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        return True, cleaned, script, None, latency_ms

