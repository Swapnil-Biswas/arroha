"""
app/guardrails/validator.py
---------------------------
Unified guardrails orchestrator providing single-point entry for:
- Input query validation
- Context grounding verification
- Output formatting & safety boundaries
"""

from __future__ import annotations

import logging
from typing import Optional

from app.guardrails.grounding import GroundingChecker
from app.guardrails.input import InputGuardrail
from app.guardrails.output import OutputGuardrail
from app.schemas.response import GroundingResult, SourceDocument

logger = logging.getLogger(__name__)


class GuardrailsValidator:
    """
    Coordinates all input, grounding, and output validation rules.
    """

    def __init__(
        self,
        input_guardrail: Optional[InputGuardrail] = None,
        grounding_checker: Optional[GroundingChecker] = None,
        output_guardrail: Optional[OutputGuardrail] = None,
    ) -> None:
        self.input_guardrail = input_guardrail or InputGuardrail()
        self.grounding_checker = grounding_checker or GroundingChecker()
        self.output_guardrail = output_guardrail or OutputGuardrail()

    def validate_input(
        self,
        query: str,
        language_hint: Optional[str] = None,
    ) -> tuple[bool, str, str, Optional[str], float]:
        """Validate raw incoming query."""
        return self.input_guardrail.validate(query, language_hint)

    def check_grounding(
        self,
        query: str,
        sources: list[SourceDocument],
        generated_answer: str,
    ) -> tuple[GroundingResult, float]:
        """Check grounding and hallucination risk."""
        return self.grounding_checker.check(query, sources, generated_answer)

    def sanitize_output(
        self,
        raw_answer: str,
        is_refusal: bool = False,
        language: str = "en",
    ) -> tuple[str, float]:
        """Sanitize and bound generated output with localized refusal."""
        return self.output_guardrail.validate_and_clean(raw_answer, is_refusal=is_refusal, language=language)
