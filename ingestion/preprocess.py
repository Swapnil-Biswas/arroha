"""
ingestion/preprocess.py
-----------------------
Multilingual text preprocessing, Unicode normalization, script detection,
and canonical document extraction from raw dataset records.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Generator, Iterable

from ingestion.models import DatasetRecord, Document


# ---------------------------------------------------------------------------
# Script Detection Map (Unicode Code Point Ranges)
# ---------------------------------------------------------------------------
SCRIPT_RANGES = {
    "Devanagari": (0x0900, 0x097F), # Hindi, Marathi, Sanskrit, Nepali
    "Bengali":    (0x0980, 0x09FF), # Bengali, Assamese
    "Gurmukhi":   (0x0A00, 0x0A7F), # Punjabi
    "Gujarati":   (0x0A80, 0x0AFF), # Gujarati
    "Oriya":      (0x0B00, 0x0B7F), # Odia
    "Tamil":      (0x0B80, 0x0BFF), # Tamil
    "Telugu":     (0x0C00, 0x0C7F), # Telugu
    "Kannada":    (0x0C80, 0x0CFF), # Kannada
    "Malayalam":  (0x0D00, 0x0D7F), # Malayalam
    "Arabic":     (0x0600, 0x06FF), # Urdu
    "Latin":      (0x0041, 0x007A), # English
}

# Regex to collapse multi-whitespace while preserving linebreaks where helpful
WHITESPACE_RE = re.compile(r"[ \t]+")
MULTILINE_RE = re.compile(r"\n{3,}")


def detect_script(text: str) -> str:
    """
    Detect the primary writing script of a text sample.
    Returns script name (e.g. 'Devanagari', 'Bengali', 'Latin', etc.)
    """
    if not text:
        return "Unknown"

    counts: dict[str, int] = {s: 0 for s in SCRIPT_RANGES}
    for ch in text:
        cp = ord(ch)
        for script, (start, end) in SCRIPT_RANGES.items():
            if start <= cp <= end:
                counts[script] += 1
                break

    best_script, max_count = max(counts.items(), key=lambda x: x[1])
    if max_count == 0:
        return "Latin"
    return best_script


def clean_multilingual_text(text: str) -> str:
    """
    Clean and normalize multilingual text:
    1. Unicode NFC normalization.
    2. Normalize common zero-width spaces / joiners / control characters.
    3. Normalize punctuation (curly quotes, dashes, Indic dandas).
    4. Strip extraneous whitespace.
    """
    if not text:
        return ""

    # 1. NFC normalization
    text = unicodedata.normalize("NFC", text)

    # 2. Remove zero-width non-joiner / joiner if malformed, replace invisible spaces
    text = text.replace("\u200b", "").replace("\ufeff", "").replace("\xa0", " ")

    # 3. Normalize quotes and dashes
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = text.replace("—", " - ").replace("–", " - ")

    # 4. Collapse spaces
    text = WHITESPACE_RE.sub(" ", text)
    text = MULTILINE_RE.sub("\n\n", text)

    return text.strip()


def extract_documents_from_record(
    record: DatasetRecord | dict,
    include_translated: bool = True,
    include_english: bool = True,
) -> list[Document]:
    """
    Extract canonical Document instances from a dataset record.

    CRITICAL DATA SAFETY:
    - Gold answers ('Answer', 'Eng_Answer') are NEVER indexed into text.
    - 'is_selected' is preserved only as evaluation metadata.
    """
    if isinstance(record, dict):
        # Normalize dict keys to lowercase where needed
        passages_dict = record.get("passages", {})
        query_id = record.get("query_id", "unknown")
        src_lang = record.get("source_lang", "en")
        tgt_lang = record.get("target_lang", "hi")
        meta = record.get("meta", {})
    else:
        passages_dict = record.passages
        query_id = record.query_id
        src_lang = record.source_lang
        tgt_lang = record.target_lang
        meta = record.meta or {}

    documents: list[Document] = []

    # Extract passage lists
    trans_passages = passages_dict.get("Translated_passages", [])
    eng_passages = passages_dict.get("English_passages", [])
    is_selected_list = passages_dict.get("is_selected", [])

    num_passages = max(len(trans_passages), len(eng_passages), len(is_selected_list))

    for idx in range(num_passages):
        selected_flag = int(is_selected_list[idx]) if idx < len(is_selected_list) else 0

        # 1. Translated passage (e.g. Hindi, Bengali, Tamil, etc.)
        if include_translated and idx < len(trans_passages):
            raw_text = str(trans_passages[idx] or "")
            cleaned_text = clean_multilingual_text(raw_text)
            if cleaned_text:
                doc_id = Document.create_id(query_id, idx, tgt_lang)
                documents.append(
                    Document(
                        id=doc_id,
                        text=cleaned_text,
                        language=tgt_lang,
                        source_lang=src_lang,
                        target_lang=tgt_lang,
                        query_id=query_id,
                        passage_id=idx,
                        is_selected=selected_flag,
                        metadata={
                            "passage_type": "translated",
                            "passage_index": idx,
                            "script": detect_script(cleaned_text),
                            **(meta if isinstance(meta, dict) else {}),
                        },
                    )
                )

        # 2. English original passage
        if include_english and idx < len(eng_passages):
            raw_text = str(eng_passages[idx] or "")
            cleaned_text = clean_multilingual_text(raw_text)
            if cleaned_text:
                doc_id = Document.create_id(query_id, idx, "en")
                documents.append(
                    Document(
                        id=doc_id,
                        text=cleaned_text,
                        language="en",
                        source_lang=src_lang,
                        target_lang=tgt_lang,
                        query_id=query_id,
                        passage_id=idx,
                        is_selected=selected_flag,
                        metadata={
                            "passage_type": "english",
                            "passage_index": idx,
                            "script": "Latin",
                            **(meta if isinstance(meta, dict) else {}),
                        },
                    )
                )

    return documents


def batch_preprocess_records(
    records: Iterable[DatasetRecord | dict],
    include_translated: bool = True,
    include_english: bool = True,
) -> Generator[Document, None, None]:
    """Stream and yield preprocessed Document instances."""
    for record in records:
        docs = extract_documents_from_record(
            record,
            include_translated=include_translated,
            include_english=include_english,
        )
        for doc in docs:
            yield doc
