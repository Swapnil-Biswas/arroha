"""
evaluation/retrieval.py
-----------------------
Retrieval quality evaluation suite.
Calculates Recall@K, Precision@K, and Mean Reciprocal Rank (MRR@K)
comparing Dense vs. BM25 vs. Hybrid retrieval strategies using MSMARCO-XI relevance labels.

Mathematical Definitions:
  - Recall@K    = (Number of unique relevant passages retrieved in top K) / (Total relevant passages)
  - Precision@K = (Number of unique relevant passages retrieved in top K) / K
  - MRR@K       = 1 / (rank of first relevant passage in top K), or 0.0 if not found in top K
All metrics are mathematically bounded strictly in [0.0, 1.0].
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

from app.retrieval.bm25 import BM25Retriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.vector import VectorRetriever
from ingestion.download import create_sample_multilingual_corpus
from ingestion.models import DatasetRecord

logger = logging.getLogger("eval_retrieval")


def evaluate_retriever(
    retriever_type: str,
    records: list[DatasetRecord],
    top_k: int = 5,
) -> dict[str, float]:
    """
    Evaluate a retriever (dense, bm25, or hybrid) on a dataset split with is_selected gold labels.
    """
    if retriever_type == "dense":
        retriever = VectorRetriever()
    elif retriever_type == "bm25":
        retriever = BM25Retriever()
    elif retriever_type == "hybrid":
        retriever = HybridRetriever()
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")

    recalls: list[float] = []
    precisions: list[float] = []
    reciprocal_ranks: list[float] = []

    for rec in records:
        query_text = rec.query
        target_qid = str(rec.query_id)

        # Identify gold relevant passage indices for this query
        passages_dict = rec.passages
        is_selected_list = passages_dict.get("is_selected", [])
        gold_passage_set = {idx for idx, s in enumerate(is_selected_list) if s == 1}

        if not gold_passage_set:
            continue

        # Execute retrieval
        if retriever_type == "dense":
            results, _, _ = retriever.search(query_text, top_k=top_k)
            retrieved_items = [(str(r[0].get("query_id")), r[0].get("passage_id")) for r in results]
        elif retriever_type == "bm25":
            results, _ = retriever.search(query_text, top_k=top_k)
            retrieved_items = [(str(r[0].get("query_id")), r[0].get("passage_id")) for r in results]
        else:  # hybrid
            results, _ = retriever.search(query_text, top_k=top_k)
            retrieved_items = [(str(r.query_id), r.passage_id) for r in results]

        # Calculate standard bounded metrics
        retrieved_gold_passages: set[int] = set()
        first_hit_rank: int | None = None

        for rank, (qid, pid) in enumerate(retrieved_items, 1):
            if pid is not None and pid in gold_passage_set and qid == target_qid:
                retrieved_gold_passages.add(pid)
                if first_hit_rank is None:
                    first_hit_rank = rank

        recall = len(retrieved_gold_passages) / len(gold_passage_set)
        precision = len(retrieved_gold_passages) / top_k
        mrr = (1.0 / first_hit_rank) if (first_hit_rank is not None and first_hit_rank <= top_k) else 0.0

        # Assertions enforcing standard metric bounds
        assert 0.0 <= recall <= 1.0, f"Invalid recall {recall} for query {target_qid}"
        assert 0.0 <= precision <= 1.0, f"Invalid precision {precision} for query {target_qid}"
        assert 0.0 <= mrr <= 1.0, f"Invalid MRR {mrr} for query {target_qid}"

        recalls.append(recall)
        precisions.append(precision)
        reciprocal_ranks.append(mrr)

    return {
        "recall_at_k": float(sum(recalls) / len(recalls)) if recalls else 0.0,
        "precision_at_k": float(sum(precisions) / len(precisions)) if precisions else 0.0,
        "mrr_at_k": float(sum(reciprocal_ranks) / len(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "evaluated_queries": len(recalls),
    }


def run_retrieval_comparison(top_k: int = 5) -> dict[str, dict[str, float]]:
    """Compare BM25 vs Dense vs Hybrid retrieval accuracy."""
    print("=" * 72)
    print("  HH Goa 2026: Retrieval Quality Comparison (Top-K = %d)" % top_k)
    print("=" * 72)

    records = create_sample_multilingual_corpus()

    print(f"\nEvaluating on {len(records)} multilingual test records...")
    bm25_metrics = evaluate_retriever("bm25", records, top_k=top_k)
    dense_metrics = evaluate_retriever("dense", records, top_k=top_k)
    hybrid_metrics = evaluate_retriever("hybrid", records, top_k=top_k)

    print("-" * 72)
    print(f"{'Retriever':<16} | {'Recall@%d' % top_k:<12} | {'Precision@%d' % top_k:<14} | {'MRR@%d' % top_k:<10}")
    print("-" * 72)
    print(f"{'BM25 (Sparse)':<16} | {bm25_metrics['recall_at_k']:>12.4f} | {bm25_metrics['precision_at_k']:>14.4f} | {bm25_metrics['mrr_at_k']:>10.4f}")
    print(f"{'FAISS (Dense)':<16} | {dense_metrics['recall_at_k']:>12.4f} | {dense_metrics['precision_at_k']:>14.4f} | {dense_metrics['mrr_at_k']:>10.4f}")
    print(f"{'Hybrid (Fused)':<16} | {hybrid_metrics['recall_at_k']:>12.4f} | {hybrid_metrics['precision_at_k']:>14.4f} | {hybrid_metrics['mrr_at_k']:>10.4f}")
    print("=" * 72)

    return {
        "bm25": bm25_metrics,
        "dense": dense_metrics,
        "hybrid": hybrid_metrics,
    }


if __name__ == "__main__":
    run_retrieval_comparison()
