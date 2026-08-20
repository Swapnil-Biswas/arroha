"""
evaluation/latency.py
---------------------
High-resolution latency benchmarking harness.
Instruments and calculates P50, P70, P100, Mean, Min, and Max latencies
across representative multilingual queries against the <200ms target.
"""

from __future__ import annotations

import logging
import statistics
import sys
import time
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np

from app.config import LATENCY_BUDGET_MS, STRETCH_LATENCY_BUDGET_MS
from app.pipeline import RAGPipeline
from app.schemas.query import QueryRequest, VoiceQueryRequest

logger = logging.getLogger("benchmark")

# Representative multilingual evaluation query suite
BENCHMARK_QUERIES = [
    # Hindi
    ("भारत की राजधानी क्या है और इसका इतिहास क्या है?", "hi"),
    ("नई दिल्ली में राष्ट्रपति भवन कहाँ स्थित है?", "hi"),
    # Bengali
    ("রবীন্দ্রনাথ ঠাকুর কে ছিলেন এবং তিনি কোন পুরস্কার পেয়েছিলেন?", "bn"),
    ("গীতাঞ্জলির জন্য রবীন্দ্রনাথ ঠাকুর কবে নোবেল পান?", "bn"),
    # Tamil
    ("தமிழ்நாட்டின் தலைநகரம் எது மற்றும் அதன் சிறப்பு என்ன?", "ta"),
    ("மதுரை நகரம் எந்த ஆற்றின் கரையில் அமைந்துள்ளது?", "ta"),
    # Marathi
    ("महाराष्ट्राची राजधानी कोणती आहे?", "mr"),
    ("पुणे शहर कशासाठी प्रसिद्ध आहे?", "mr"),
    # Telugu
    ("హైదరాబాద్ నగరం యొక్క ప్రాముఖ్యత ఏమిటి?", "te"),
    ("చార్మినార్ ఎప్పుడు నిర్మించబడింది?", "te"),
    # Gujarati
    ("ગુજરાતનું સૌથી મોટું શહેર કયું છે?", "gu"),
    # English
    ("How does Retrieval-Augmented Generation work in voice pipelines?", "en"),
    ("What is Automatic Speech Recognition in voice AI systems?", "en"),
    # Out of domain / refusal test
    ("Who won the 1974 Antarctic soccer championship?", "en"),
]


def calculate_percentiles(values: list[float]) -> dict[str, float]:
    """Calculate P50, P70, P90, P100, Mean, Min, and Max from a list of values."""
    if not values:
        return {}
    arr = np.array(values)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p70": float(np.percentile(arr, 70)),
        "p90": float(np.percentile(arr, 90)),
        "p100": float(np.percentile(arr, 100)),
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
    }


def run_latency_benchmark(num_runs: int = 15) -> dict[str, Any]:
    """
    Run comprehensive latency benchmarks across test queries and compute metrics.
    """
    print("=" * 78)
    print("  HH Goa 2026: Multilingual Voice RAG Latency Benchmark")
    print(f"  Target Budget : < {LATENCY_BUDGET_MS:.1f} ms")
    print(f"  Stretch Target: < {STRETCH_LATENCY_BUDGET_MS:.1f} ms")
    print(f"  Test Suite    : {len(BENCHMARK_QUERIES)} multilingual queries x {num_runs} iterations")
    print("=" * 78)

    pipeline = RAGPipeline()

    # Warmup pipeline (initializes models/FAISS in memory)
    print("\nWarming up models and index caches...")
    warmup_req = QueryRequest(query="भारत की राजधानी", language="hi")
    pipeline.process_query(warmup_req)
    print("Warmup complete. Starting benchmark runs...\n")

    stage_latencies: dict[str, list[float]] = {
        "input_guardrails": [],
        "query_embed": [],
        "vector_retrieval": [],
        "bm25_retrieval": [],
        "hybrid_fusion": [],
        "reranker": [],
        "prompt_construction": [],
        "llm_generation": [],
        "grounding_check": [],
        "total": [],
    }

    all_responses = []

    for run_idx in range(num_runs):
        for query_text, lang in BENCHMARK_QUERIES:
            req = QueryRequest(query=query_text, language=lang)
            res = pipeline.process_query(req)
            all_responses.append(res)

            l = res.latency
            stage_latencies["input_guardrails"].append(l.input_guardrails_ms)
            stage_latencies["query_embed"].append(l.query_embed_ms)
            stage_latencies["vector_retrieval"].append(l.vector_retrieval_ms)
            stage_latencies["bm25_retrieval"].append(l.bm25_retrieval_ms)
            stage_latencies["hybrid_fusion"].append(l.hybrid_fusion_ms)
            stage_latencies["reranker"].append(l.reranker_ms)
            stage_latencies["prompt_construction"].append(l.prompt_construction_ms)
            stage_latencies["llm_generation"].append(l.llm_generation_ms)
            stage_latencies["grounding_check"].append(l.grounding_check_ms)
            stage_latencies["total"].append(l.total_ms)

    # Print Summary Table
    print("-" * 78)
    print(f"{'Pipeline Stage':<24} | {'Mean (ms)':<9} | {'P50 (ms)':<8} | {'P70 (ms)':<8} | {'P100 (ms)':<9}")
    print("-" * 78)

    summary_metrics = {}
    for stage, vals in stage_latencies.items():
        stats = calculate_percentiles(vals)
        summary_metrics[stage] = stats
        stage_name = stage.replace("_", " ").title()
        print(f"{stage_name:<24} | {stats['mean']:>9.2f} | {stats['p50']:>8.2f} | {stats['p70']:>8.2f} | {stats['p100']:>9.2f}")

    print("-" * 78)
    total_stats = summary_metrics["total"]
    print(f"{'OVERALL PIPELINE':<24} | {total_stats['mean']:>9.2f} | {total_stats['p50']:>8.2f} | {total_stats['p70']:>8.2f} | {total_stats['p100']:>9.2f}")
    print("=" * 78)

    # Verification against target budget
    p50 = total_stats["p50"]
    p70 = total_stats["p70"]
    p100 = total_stats["p100"]

    p50_pass = p50 <= LATENCY_BUDGET_MS
    p70_pass = p70 <= LATENCY_BUDGET_MS
    p100_pass = p100 <= LATENCY_BUDGET_MS

    print("\n--- Latency Budget Compliance ---")
    print(f"  P50  : {p50:.2f} ms [{'PASS' if p50_pass else 'FAIL'}] (Target: < {LATENCY_BUDGET_MS:.0f} ms)")
    print(f"  P70  : {p70:.2f} ms [{'PASS' if p70_pass else 'FAIL'}] (Target: < {LATENCY_BUDGET_MS:.0f} ms)")
    print(f"  P100 : {p100:.2f} ms [{'PASS' if p100_pass else 'FAIL'}] (Target: < {LATENCY_BUDGET_MS:.0f} ms)")

    stretch_p50 = p50 <= STRETCH_LATENCY_BUDGET_MS
    print(f"  Stretch Goal (< 150 ms P50): [{'ACHIEVED' if stretch_p50 else 'NOT YET ACHIEVED'}]")
    print("=" * 78)

    return summary_metrics


if __name__ == "__main__":
    run_latency_benchmark(num_runs=5)
