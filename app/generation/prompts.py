"""
app/generation/prompts.py
-------------------------
Multilingual system prompts and context assembly templates.
Strictly enforces grounding, concise answers (<60 words), and language preservation.
"""

from __future__ import annotations

from app.schemas.response import SourceDocument

SYSTEM_PROMPT = """You are a multilingual factual AI assistant for a real-time voice pipeline.
Answer the user's question accurately and concisely using ONLY the provided retrieved context.

CRITICAL RULES:
1. Grounding: Answer strictly using facts from the retrieved context. Do NOT extrapolate, speculate, or use outside knowledge.
2. Refusal: If the retrieved context does not contain enough information to answer the question, state clearly: "I do not have enough information in the retrieved sources to answer this question." (or its equivalent in the query language).
3. Language Consistency: Reply in the same language and script as the user's query (e.g. Hindi in Devanagari, Bengali in Bengali script, Tamil in Tamil script, English in Latin).
4. Conciseness: Keep the answer under 2-3 sentences (maximum 50 words) to ensure low latency for voice synthesis.
5. No Meta-Commentary: Do NOT say "Based on the provided text" or "According to the context". State the factual answer directly.
"""


def build_rag_prompt(
    query: str,
    sources: list[SourceDocument],
    max_context_tokens: int = 600,
) -> tuple[str, str]:
    """
    Format system and user messages with retrieved context snippets.
    Returns (system_prompt, user_message).
    """
    if not sources:
        user_message = f"Retrieved Context:\n[NO RELEVANT CONTEXT FOUND]\n\nUser Question: {query}"
        return SYSTEM_PROMPT, user_message

    context_snippets: list[str] = []
    total_chars = 0
    max_chars = max_context_tokens * 4  # Approx 4 chars per token

    for idx, doc in enumerate(sources, 1):
        clean_text = doc.text.strip()
        if total_chars + len(clean_text) > max_chars and context_snippets:
            break
        context_snippets.append(f"[Source {idx} - Lang: {doc.language}]: {clean_text}")
        total_chars += len(clean_text)

    context_block = "\n\n".join(context_snippets)

    user_message = (
        f"Retrieved Context:\n"
        f"{context_block}\n\n"
        f"User Question: {query}\n\n"
        f"Factual Answer:"
    )

    return SYSTEM_PROMPT, user_message
