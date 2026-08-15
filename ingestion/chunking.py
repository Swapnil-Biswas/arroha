"""
ingestion/chunking.py
---------------------
Pluggable chunking strategies for multilingual documents.

Implemented Strategies:
  1. FixedSizeChunker: Fixed character/token windows with overlap.
  2. SentenceAwareChunker: Multilingual sentence boundary splitting (supports Indic । and latin . ? !).
  3. PassageAwareChunker: Preserves passage boundary integrity while respecting max size.
  4. RecursiveChunker: Hierarchical recursive splitting (paragraphs -> sentences -> words).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Generator, Iterable

from ingestion.models import Chunk, Document

# Multilingual sentence boundary regex
# Matches . ! ? and Indic danda (।) and double danda (॥), followed by space or end of string
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।॥\n])\s+")


class BaseChunker(ABC):
    """Abstract base class for document chunkers."""

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = max(0, min(chunk_overlap, chunk_size // 2))

    @abstractmethod
    def chunk_document(self, doc: Document) -> list[Chunk]:
        """Split a single Document into a list of Chunks."""
        raise NotImplementedError

    def chunk_documents(self, docs: Iterable[Document]) -> Generator[Chunk, None, None]:
        """Chunk a stream of documents."""
        for doc in docs:
            chunks = self.chunk_document(doc)
            for chunk in chunks:
                yield chunk


class FixedSizeChunker(BaseChunker):
    """
    Fixed-size character chunking with configurable sliding-window overlap.
    """

    def chunk_document(self, doc: Document) -> list[Chunk]:
        text = doc.text
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [
                Chunk(
                    chunk_id=Chunk.create_id(doc.id, 0),
                    doc_id=doc.id,
                    text=text,
                    language=doc.language,
                    chunk_index=0,
                    start_char=0,
                    end_char=len(text),
                    query_id=doc.query_id,
                    passage_id=doc.passage_id,
                    is_selected=doc.is_selected,
                    metadata=doc.metadata,
                )
            ]

        chunks: list[Chunk] = []
        start = 0
        idx = 0
        step = self.chunk_size - self.chunk_overlap

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    Chunk(
                        chunk_id=Chunk.create_id(doc.id, idx),
                        doc_id=doc.id,
                        text=chunk_text,
                        language=doc.language,
                        chunk_index=idx,
                        start_char=start,
                        end_char=end,
                        query_id=doc.query_id,
                        passage_id=doc.passage_id,
                        is_selected=doc.is_selected,
                        metadata=doc.metadata,
                    )
                )
                idx += 1

            if end >= len(text):
                break
            start += step

        return chunks


class SentenceAwareChunker(BaseChunker):
    """
    Sentence-aware chunking for multilingual texts.
    Splits along sentence boundaries (Latin punctuation & Indic dandas ।/॥)
    and groups sentences up to chunk_size characters.
    """

    def chunk_document(self, doc: Document) -> list[Chunk]:
        text = doc.text
        if not text:
            return []

        # Split into sentences
        sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
        if not sentences:
            return []

        chunks: list[Chunk] = []
        current_sentences: list[str] = []
        current_len = 0
        chunk_idx = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # If adding this sentence exceeds chunk_size and we already have accumulated sentences
            if current_len + sentence_len + 1 > self.chunk_size and current_sentences:
                chunk_text = " ".join(current_sentences).strip()
                chunks.append(
                    Chunk(
                        chunk_id=Chunk.create_id(doc.id, chunk_idx),
                        doc_id=doc.id,
                        text=chunk_text,
                        language=doc.language,
                        chunk_index=chunk_idx,
                        query_id=doc.query_id,
                        passage_id=doc.passage_id,
                        is_selected=doc.is_selected,
                        metadata={**doc.metadata, "strategy": "sentence"},
                    )
                )
                chunk_idx += 1

                # Handle overlap: retain last sentence if it's smaller than overlap budget
                if self.chunk_overlap > 0 and len(current_sentences[-1]) <= self.chunk_overlap:
                    current_sentences = [current_sentences[-1], sentence]
                    current_len = len(current_sentences[0]) + sentence_len + 1
                else:
                    current_sentences = [sentence]
                    current_len = sentence_len
            else:
                current_sentences.append(sentence)
                current_len += sentence_len + 1

        # Emit leftover sentences
        if current_sentences:
            chunk_text = " ".join(current_sentences).strip()
            chunks.append(
                Chunk(
                    chunk_id=Chunk.create_id(doc.id, chunk_idx),
                    doc_id=doc.id,
                    text=chunk_text,
                    language=doc.language,
                    chunk_index=chunk_idx,
                    query_id=doc.query_id,
                    passage_id=doc.passage_id,
                    is_selected=doc.is_selected,
                    metadata={**doc.metadata, "strategy": "sentence"},
                )
            )

        return chunks


class PassageAwareChunker(BaseChunker):
    """
    Metadata-aware chunker that preserves MSMARCO passage integrity.
    If passage length is <= chunk_size, keeps the passage atomic.
    Otherwise delegates to sentence splitting.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._fallback = SentenceAwareChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def chunk_document(self, doc: Document) -> list[Chunk]:
        text = doc.text.strip()
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [
                Chunk(
                    chunk_id=Chunk.create_id(doc.id, 0),
                    doc_id=doc.id,
                    text=text,
                    language=doc.language,
                    chunk_index=0,
                    start_char=0,
                    end_char=len(text),
                    query_id=doc.query_id,
                    passage_id=doc.passage_id,
                    is_selected=doc.is_selected,
                    metadata={**doc.metadata, "strategy": "passage_atomic"},
                )
            ]

        return self._fallback.chunk_document(doc)


class RecursiveChunker(BaseChunker):
    """
    Hierarchical recursive chunker splitting on double newline,
    single newline, sentence boundaries, and word boundaries.
    """

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50) -> None:
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.separators = ["\n\n", "\n", "। ", ". ", "? ", "! ", " "]

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]

        sep = separators[0]
        splits = text.split(sep)
        result: list[str] = []
        current = ""

        for part in splits:
            part_str = part.strip()
            if not part_str:
                continue

            test = f"{current}{sep}{part_str}" if current else part_str
            if len(test) <= self.chunk_size:
                current = test
            else:
                if current:
                    result.append(current)
                if len(part_str) > self.chunk_size:
                    # Recurse with finer separator
                    result.extend(self._split_text(part_str, separators[1:]))
                    current = ""
                else:
                    current = part_str

        if current:
            result.append(current)
        return result

    def chunk_document(self, doc: Document) -> list[Chunk]:
        text = doc.text.strip()
        if not text:
            return []

        raw_chunks = self._split_text(text, self.separators)
        chunks: list[Chunk] = []

        for idx, chunk_text in enumerate(raw_chunks):
            cleaned = chunk_text.strip()
            if cleaned:
                chunks.append(
                    Chunk(
                        chunk_id=Chunk.create_id(doc.id, idx),
                        doc_id=doc.id,
                        text=cleaned,
                        language=doc.language,
                        chunk_index=idx,
                        query_id=doc.query_id,
                        passage_id=doc.passage_id,
                        is_selected=doc.is_selected,
                        metadata={**doc.metadata, "strategy": "recursive"},
                    )
                )

        return chunks


def get_chunker(
    strategy: str = "sentence",
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> BaseChunker:
    """Factory function to instantiate selected chunking strategy."""
    strategy_lower = strategy.lower().strip()
    if strategy_lower == "fixed":
        return FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif strategy_lower == "sentence":
        return SentenceAwareChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif strategy_lower == "passage":
        return PassageAwareChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif strategy_lower == "recursive":
        return RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        raise ValueError(f"Unknown chunking strategy '{strategy}'. Choose from: fixed, sentence, passage, recursive.")
