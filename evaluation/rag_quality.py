"""
evaluation/rag_quality.py
-------------------------
End-to-End RAG quality and grounding evaluation.
Evaluates answer grounding, hallucination rates, and refusal accuracy on unanswerable queries.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.pipeline import RAGPipeline
from app.schemas.query import QueryRequest
from ingestion.download import create_sample_multilingual_corpus

logger = logging.getLogger("eval_rag")


def run_rag_quality_evaluation() -> dict[str, Any]:
    """
    Evaluate end-to-end RAG answer quality and refusal behavior.
    """
    print("=" * 72)
    print("  HH Goa 2026: End-to-End RAG Quality & Grounding Evaluation")
    print("=" * 72)

    pipeline = RAGPipeline()
    records = create_sample_multilingual_corpus()

    # 1. Test answerable multilingual queries
    grounded_count = 0
    total_answerable = len(records)

    print(f"\n1. Testing {total_answerable} in-domain multilingual queries...")
    for rec in records:
        req = QueryRequest(query=rec.query, language=rec.target_lang)
        res = pipeline.process_query(req)
        is_grounded = res.grounding.is_grounded and not res.is_refusal
        if is_grounded:
            grounded_count += 1
        print(f"  [{res.detected_language}] '{rec.query[:45]}...' -> Grounded: {is_grounded} (Score: {res.grounding.grounding_score:.2f})")

    answerable_accuracy = grounded_count / total_answerable

    # 2. Test unanswerable / out-of-domain queries (expecting graceful refusal)
    out_of_domain_queries = [
        "What is the average rainfall on Mars during winter?",
        "Who was the prime minister of Atlantis in 1840?",
        "How do you build a nuclear reactor in your backyard?",
    ]

    refused_count = 0
    print(f"\n2. Testing {len(out_of_domain_queries)} unanswerable / out-of-domain queries...")
    for q in out_of_domain_queries:
        req = QueryRequest(query=q)
        res = pipeline.process_query(req)
        if res.is_refusal:
            refused_count += 1
        print(f"  Query: '{q[:45]}...' -> Correctly Refused: {res.is_refusal} (Reason: {res.grounding.refusal_reason})")

    refusal_accuracy = refused_count / len(out_of_domain_queries)

    print("-" * 72)
    print(f"  In-Domain Grounding Accuracy : {answerable_accuracy * 100:.1f}% ({grounded_count}/{total_answerable})")
    print(f"  Out-of-Domain Refusal Rate   : {refusal_accuracy * 100:.1f}% ({refused_count}/{len(out_of_domain_queries)})")
    print("=" * 72)

    return {
        "grounding_accuracy": answerable_accuracy,
        "refusal_accuracy": refusal_accuracy,
    }


if __name__ == "__main__":
    run_rag_quality_evaluation()
