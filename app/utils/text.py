r"""
app/utils/text.py
-----------------
Script-aware tokenization utility for ARROHA Multilingual Voice RAG.
Preserves full Devanagari and Indic words (vowel signs, virama, matras)
by splitting on whitespace and punctuation separators rather than standard ASCII character classes (\w).
"""


from __future__ import annotations

import re

# Separator regex: whitespace, Devanagari danda (U+0964), double danda (U+0965), and ASCII punctuation
_SEPARATORS = re.compile(r"[\s।॥!-/:-@\[-`{-~‐-‧‰-⁞]+")
MIN_TOKEN_LEN = 2


def tokenize(text: str, *, lower: bool = True, min_len: int = MIN_TOKEN_LEN) -> list[str]:
    """Split on separators, preserving whole words in any Indic or Latin script."""
    if not text:
        return []
    s = text.lower() if lower else text
    return [t for t in _SEPARATORS.split(s) if len(t) >= min_len]


def token_set(text: str, *, min_len: int = MIN_TOKEN_LEN) -> set[str]:
    """Return unique set of tokens from text."""
    return set(tokenize(text, min_len=min_len))


def token_overlap(query: str, candidate: str, *, min_len: int = MIN_TOKEN_LEN) -> float:
    """
    Fraction of the query's tokens present in the candidate.
    Recall-oriented score: measures coverage of query terms in candidate.
    """
    q_tokens = token_set(query, min_len=min_len)
    if not q_tokens:
        return 0.0
    cand_tokens = token_set(candidate, min_len=min_len)
    return len(q_tokens & cand_tokens) / len(q_tokens)
