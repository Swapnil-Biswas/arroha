"""
ingestion/models.py
-------------------
Canonical data models for dataset records, documents, and chunks.
Enforces data-safety rules: gold answers are never placed into searchable text.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional
from pydantic import BaseModel, Field


class DatasetRecord(BaseModel):
    """Raw record representation from ai4bharat/MSMARCO-XI."""
    query_id: str | int
    query_type: Optional[str] = None
    query: str                       # Translated query
    eng_query: Optional[str] = None  # Original English query
    answer: Optional[str] = None     # Translated gold answer (DO NOT INDEX)
    eng_answer: Optional[str] = None # English gold answer (DO NOT INDEX)
    source_lang: str = "en"
    target_lang: str = "hi"
    passages: dict[str, list[Any]] = Field(default_factory=dict)
    # Expected passages keys: 'English_passages', 'Translated_passages', 'is_selected'
    meta: Optional[dict[str, Any]] = None


class Document(BaseModel):
    """
    Canonical searchable document.
    Represents an individual passage extracted from a dataset record.
    """
    id: str = Field(..., description="Deterministic document unique identifier")
    text: str = Field(..., description="Cleaned, searchable passage text")
    language: str = Field(..., description="Language code of text (e.g. 'hin', 'ben', 'eng')")
    source_lang: str = "en"
    target_lang: str = "hi"
    query_id: str | int
    passage_id: int
    is_selected: int = Field(0, description="Gold relevance label: 1 if relevant to query, 0 otherwise")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def create_id(cls, query_id: str | int, passage_id: int, lang: str) -> str:
        """Create a deterministic unique document ID."""
        raw = f"{query_id}_{passage_id}_{lang}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class Chunk(BaseModel):
    """A chunked segment of a canonical document."""
    chunk_id: str
    doc_id: str
    text: str
    language: str
    chunk_index: int
    start_char: int = 0
    end_char: int = 0
    query_id: Optional[str | int] = None
    passage_id: Optional[int] = None
    is_selected: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def create_id(cls, doc_id: str, chunk_index: int) -> str:
        return f"{doc_id}_c{chunk_index}"
