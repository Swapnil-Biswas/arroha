"""
app/generation/prompts.py
-------------------------
Multilingual system prompts and context assembly templates for ARROHA.
Strictly enforces:
1. Grounding from retrieved context only.
2. Exact language & script preservation (Bengali -> Bengali, Hindi -> Hindi, etc.).
3. 1-2 complete, concise sentences without mid-response truncation.
"""

from __future__ import annotations

from typing import Optional

from app.schemas.response import SourceDocument
from ingestion.preprocess import detect_script

LANGUAGE_METADATA: dict[str, tuple[str, str]] = {
    "en": ("English", "Latin"),
    "hi": ("Hindi", "Devanagari (हिन्दी)"),
    "bn": ("Bengali", "Bengali (বাংলা)"),
    "ta": ("Tamil", "Tamil (தமிழ்)"),
    "te": ("Telugu", "Telugu (తెలుగు)"),
    "mr": ("Marathi", "Devanagari (मराठी)"),
    "gu": ("Gujarati", "Gujarati (ગુજરાતી)"),
    "kn": ("Kannada", "Kannada (ಕನ್ನಡ)"),
    "ml": ("Malayalam", "Malayalam (മലയാളം)"),
    "pa": ("Punjabi", "Gurmukhi (ਪੰਜਾਬੀ)"),
    "or": ("Odia", "Odia (ଓଡ଼ିଆ)"),
    "as": ("Assamese", "Bengali/Assamese (অসমীয়া)"),
    "ne": ("Nepali", "Devanagari (नेपाली)"),
    "sa": ("Sanskrit", "Devanagari (संस्कृतम्)"),
    "ur": ("Urdu", "Arabic/Urdu (اردو)"),
}

SCRIPT_TO_LANG_CODE: dict[str, str] = {
    "Bengali": "bn",
    "Devanagari": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Gujarati": "gu",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Gurmukhi": "pa",
    "Oriya": "or",
    "Arabic": "ur",
    "Latin": "en",
}


def resolve_query_language(query: str, language_hint: Optional[str] = None) -> tuple[str, str, str]:
    """
    Resolve ISO language code, display name, and script name.
    Returns (lang_code, language_name, script_name)
    """
    if language_hint and language_hint in LANGUAGE_METADATA:
        lang_code = language_hint
    else:
        script = detect_script(query)
        lang_code = SCRIPT_TO_LANG_CODE.get(script, "en")

    lang_name, script_name = LANGUAGE_METADATA.get(lang_code, ("English", "Latin"))
    return lang_code, lang_name, script_name


def build_rag_prompt(
    query: str,
    sources: list[SourceDocument],
    language_hint: Optional[str] = None,
    max_context_tokens: int = 250,
) -> tuple[str, str, str]:
    """
    Format system and user messages with retrieved context snippets and language directives.
    Returns: (system_prompt, user_message, detected_lang_code)
    """
    lang_code, lang_name, script_name = resolve_query_language(query, language_hint)

    system_prompt = (
        f"You are a strictly grounded multilingual factual assistant.\n"
        f"Answer the user's question accurately using ONLY facts explicitly stated in the Retrieved Context.\n\n"
        f"CRITICAL CONSTRAINTS:\n"
        f"1. Zero Hallucination: State ONLY facts explicitly present in the context below. Never use external or parametric knowledge.\n"
        f"2. Strict Refusal: If the Retrieved Context does NOT explicitly contain the factual answer to the question, you MUST output ONLY: [INSUFFICIENT_CONTEXT]\n"
        f"3. Language & Script: The user's question is in {lang_name}. State the answer strictly in {lang_name} using {script_name} script.\n"
        f"4. Direct & Accurate: State the primary entity name and factual answer directly in 1 complete sentence. Do not omit the subject or repeat system rules."
    )

    if not sources:
        user_message = (
            f"Retrieved Context:\n[NO RELEVANT CONTEXT FOUND]\n\n"
            f"User Question: {query}\n\n"
            f"Factual Answer:"
        )
        return system_prompt, user_message, lang_code

    context_snippets: list[str] = []
    total_chars = 0
    max_chars = max_context_tokens * 4  # Approx 4 chars per token

    for idx, doc in enumerate(sources, 1):
        clean_text = doc.text.strip()
        if total_chars + len(clean_text) > max_chars and context_snippets:
            break
        context_snippets.append(f"[Source {idx}]: {clean_text}")
        total_chars += len(clean_text)

    context_block = "\n\n".join(context_snippets)

    user_message = (
        f"Retrieved Context:\n"
        f"{context_block}\n\n"
        f"User Question: {query}\n\n"
        f"Factual Answer (strictly in {lang_name} using {script_name} script):"
    )

    return system_prompt, user_message, lang_code
