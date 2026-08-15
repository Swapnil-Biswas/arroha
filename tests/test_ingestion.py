"""
tests/test_ingestion.py
-----------------------
Unit tests for data models, multilingual preprocessing, normalization,
data-safety rules (no gold answers in corpus), and chunking strategies.
"""

import pytest
from ingestion.chunking import (
    FixedSizeChunker,
    PassageAwareChunker,
    RecursiveChunker,
    SentenceAwareChunker,
    get_chunker,
)
from ingestion.models import Chunk, DatasetRecord, Document
from ingestion.preprocess import (
    clean_multilingual_text,
    detect_script,
    extract_documents_from_record,
)


def test_clean_multilingual_text():
    raw = "  भारत  की \u200b राजधानी   नई  दिल्ली\xa0है।   "
    cleaned = clean_multilingual_text(raw)
    assert "\u200b" not in cleaned
    assert "\xa0" not in cleaned
    assert "  " not in cleaned
    assert cleaned == "भारत की राजधानी नई दिल्ली है।"


def test_detect_script():
    assert detect_script("भारत की राजधानी") == "Devanagari"
    assert detect_script("কলকাতা পশ্চিমবঙ্গ") == "Bengali"
    assert detect_script("தமிழ்நாடு சென்னை") == "Tamil"
    assert detect_script("హైదరాబాద్ నగరం") == "Telugu"
    assert detect_script("અમદાવાદ ગુજરાત") == "Gujarati"
    assert detect_script("What is the capital?") == "Latin"


def test_extract_documents_no_answer_leakage():
    record = DatasetRecord(
        query_id=999,
        query="भारत की राजधानी क्या है?",
        eng_query="What is the capital of India?",
        answer="SECRET_GOLD_HINDI_ANSWER",
        eng_answer="SECRET_GOLD_ENG_ANSWER",
        source_lang="en",
        target_lang="hi",
        passages={
            "English_passages": ["New Delhi is the official capital of India."],
            "Translated_passages": ["नई दिल्ली भारत की आधिकारिक राजधानी है।"],
            "is_selected": [1],
        },
    )

    docs = extract_documents_from_record(record)
    assert len(docs) == 2  # 1 translated + 1 english

    for doc in docs:
        # Verify gold answers are NEVER present in searchable document text
        assert "SECRET_GOLD_HINDI_ANSWER" not in doc.text
        assert "SECRET_GOLD_ENG_ANSWER" not in doc.text
        assert doc.query_id == 999
        assert doc.is_selected == 1


def test_chunking_strategies():
    doc = Document(
        id="test_doc_1",
        text="नई दिल्ली भारत की राजधानी है। यह एक ऐतिहासिक शहर है। यहाँ कई प्रसिद्ध स्मारक हैं।",
        language="hi",
        query_id=1,
        passage_id=0,
    )

    # 1. Fixed Chunker
    fixed = FixedSizeChunker(chunk_size=30, chunk_overlap=5)
    chunks_fixed = fixed.chunk_document(doc)
    assert len(chunks_fixed) >= 2
    assert all(isinstance(c, Chunk) for c in chunks_fixed)

    # 2. Sentence Chunker
    sentence = SentenceAwareChunker(chunk_size=40, chunk_overlap=0)
    chunks_sent = sentence.chunk_document(doc)
    assert len(chunks_sent) >= 2

    # 3. Passage Chunker
    passage = PassageAwareChunker(chunk_size=500)
    chunks_pass = passage.chunk_document(doc)
    assert len(chunks_pass) == 1  # Fits in 500 chars

    # 4. Recursive Chunker
    rec = RecursiveChunker(chunk_size=35, chunk_overlap=5)
    chunks_rec = rec.chunk_document(doc)
    assert len(chunks_rec) >= 2

    # 5. Factory
    ch = get_chunker("sentence")
    assert isinstance(ch, SentenceAwareChunker)
