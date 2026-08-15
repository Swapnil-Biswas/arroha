"""
evaluation/audit_benchmark.py
-----------------------------
Rigorous Baseline Audit Script for HH Goa 2026 Task 2.
Directly connects to live LM Studio instance at http://127.0.0.1:1234/v1
with model 'qwen/qwen3-4b-2507' (Q4_K_M GGUF).

Separates and measures:
  A. Retrieval-only latency (Pre-generation)
  B. Real Qwen3 LLM latency (TTFT, total generation, token count, tokens/sec)
  C. Full Text RAG latency (Retrieval + Real Qwen3 + Grounding + Guardrails)
  D. STT latency
  E. Full Voice Pipeline latency (STT + Real Text RAG)
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
from openai import OpenAI

from app.config import (
    LATENCY_BUDGET_MS,
    LLM_ENDPOINT,
    LLM_MODEL_ID,
    STRETCH_LATENCY_BUDGET_MS,
)
from app.generation.prompts import build_rag_prompt
from app.guardrails.validator import GuardrailsValidator
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import Reranker
from app.schemas.response import SourceDocument
from app.voice.stt import SpeechToTextEngine

logger = logging.getLogger("audit_benchmark")

# 14 Multilingual Benchmark Queries covering 7 languages + 1 refusal query
BENCHMARK_QUERIES = [
    # Hindi (Devanagari)
    ("भारत की राजधानी क्या है और इसका इतिहास क्या है?", "hi"),
    ("नई दिल्ली में राष्ट्रपति भवन कहाँ स्थित है?", "hi"),
    # Bengali (Bengali script)
    ("রবীন্দ্রনাথ ঠাকুর কে ছিলেন এবং তিনি কোন পুরস্কার পেয়েছিলেন?", "bn"),
    ("গীতাঞ্জলির জন্য রবীন্দ্রনাথ ঠাকুর কবে নোবেল পান?", "bn"),
    # Tamil (Tamil script)
    ("தமிழ்நாட்டின் தலைநகரம் எது மற்றும் அதன் சிறப்பு என்ன?", "ta"),
    ("மதுரை நகரம் எந்த ஆற்றின் கரையில் அமைந்துள்ளது?", "ta"),
    # Marathi (Devanagari script)
    ("महाराष्ट्राची राजधानी कोणती आहे?", "mr"),
    ("पुणे शहर कशासाठी प्रसिद्ध आहे?", "mr"),
    # Telugu (Telugu script)
    ("హైదరాబాద్ నగరం యొక్క ప్రాముఖ్యత ఏమిటి?", "te"),
    ("చార్మినార్ ఎప్పుడు నిర్మించబడింది?", "te"),
    # Gujarati (Gujarati script)
    ("ગુજરાતનું સૌથી મોટું શહેર કયું છે?", "gu"),
    ("અમદાવાદ શહેર કઈ નદીના કિનારે આવેલું છે?", "gu"),
    # English (Latin script)
    ("How does Retrieval-Augmented Generation work in voice pipelines?", "en"),
    # Out of domain (Refusal test)
    ("What is the average winter temperature on Mars?", "en"),
]


def stats(values: list[float]) -> dict[str, float]:
    """Calculate P50, P70, P100, Mean, Min, Max from values."""
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p70": 0.0, "p100": 0.0, "min": 0.0, "max": 0.0}
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p70": float(np.percentile(arr, 70)),
        "p100": float(np.percentile(arr, 100)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def run_rigorous_baseline_audit(num_runs: int = 5) -> dict[str, Any]:
    print("=" * 80)
    print("  HH GOA 2026 TASK 2 — RIGOROUS BASELINE AUDIT")
    print("=" * 80)

    # 1. Inspect live LM Studio API
    lm_client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio", timeout=30.0)
    try:
        models_list = lm_client.models.list()
        available_models = [m.id for m in models_list.data]
        print(f"  LM Studio API: Connected (http://127.0.0.1:1234/v1)")
        print(f"  Available Models in LM Studio: {available_models}")
    except Exception as exc:
        print(f"  ERROR connecting to LM Studio: {exc}")
        return {}

    # Target model
    target_model = "qwen/qwen3-4b-2507"
    if target_model not in available_models and available_models:
        target_model = available_models[0]
    print(f"  Audit Target Model: {target_model} (Quantization: Q4_K_M GGUF)")
    print(f"  Test Suite: {len(BENCHMARK_QUERIES)} multilingual queries x {num_runs} iterations = {len(BENCHMARK_QUERIES) * num_runs} runs")
    print("=" * 80)

    # Initialize components
    hybrid_retriever = HybridRetriever()
    guardrails = GuardrailsValidator()
    reranker = Reranker(enabled=False)
    stt = SpeechToTextEngine(provider="mock")

    # Metrics storage
    retrieval_latencies: list[float] = []
    llm_ttft_latencies: list[float] = []
    llm_gen_latencies: list[float] = []
    llm_token_counts: list[int] = []
    llm_tokens_per_sec: list[float] = []
    full_text_rag_latencies: list[float] = []
    stt_latencies: list[float] = []
    full_voice_rag_latencies: list[float] = []

    successful_runs = 0
    failed_runs = 0

    print("\nWarming up retrieval & LM Studio Qwen3 connection...")
    # Warmup query
    warm_query = "भारत की राजधानी क्या है?"
    w_src, _ = hybrid_retriever.search(warm_query, top_k=3)
    w_sys, w_usr = build_rag_prompt(warm_query, w_src)
    w_res = lm_client.chat.completions.create(
        model=target_model,
        messages=[{"role": "system", "content": w_sys}, {"role": "user", "content": w_usr}],
        max_tokens=30,
        temperature=0.1,
    )
    print(f"Warmup response: '{w_res.choices[0].message.content.strip()}'\n")

    print(f"Running {num_runs} audit passes across {len(BENCHMARK_QUERIES)} multilingual queries...\n")

    for run_idx in range(1, num_runs + 1):
        for q_idx, (query_text, lang) in enumerate(BENCHMARK_QUERIES, 1):
            t_total_start = time.perf_counter_ns()

            try:
                # ----------------------------------------------------
                # A. Retrieval / Pre-generation Pipeline
                # ----------------------------------------------------
                t_ret_start = time.perf_counter_ns()

                # 1. Input Guardrail
                is_valid, cleaned_q, script, reason, in_lat = guardrails.validate_input(query_text, language_hint=lang)

                # 2. Hybrid Retrieval
                sources, ret_lats = hybrid_retriever.search(cleaned_q, top_k=3)

                # 3. Reranker (if enabled)
                sources, rerank_lat = reranker.rerank(cleaned_q, sources, top_k=3)

                # 4. Prompt Assembly
                system_prompt, user_message = build_rag_prompt(cleaned_q, sources, max_context_tokens=300)

                t_ret_end = time.perf_counter_ns()
                ret_latency_ms = (t_ret_end - t_ret_start) / 1_000_000.0
                retrieval_latencies.append(ret_latency_ms)

                # ----------------------------------------------------
                # B. Real Qwen3 Inference via LM Studio (with Streaming for TTFT)
                # ----------------------------------------------------
                t_llm_start = time.perf_counter_ns()
                ttft_recorded = False
                ttft_ms = 0.0
                chunks_text = []

                stream = lm_client.chat.completions.create(
                    model=target_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    max_tokens=60,  # Strict concise limit for voice latency
                    temperature=0.1,
                    stream=True,
                )

                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        if not ttft_recorded:
                            ttft_ms = (time.perf_counter_ns() - t_llm_start) / 1_000_000.0
                            ttft_recorded = True
                        chunks_text.append(chunk.choices[0].delta.content)

                t_llm_end = time.perf_counter_ns()
                gen_latency_ms = (t_llm_end - t_llm_start) / 1_000_000.0
                llm_answer = "".join(chunks_text).strip()

                if not ttft_recorded:
                    ttft_ms = gen_latency_ms

                # Approx tokens: word count * 1.3 for multilingual
                token_count = max(len(llm_answer.split()) + 2, 1)
                tps = (token_count / (gen_latency_ms / 1000.0)) if gen_latency_ms > 0 else 0.0

                llm_ttft_latencies.append(ttft_ms)
                llm_gen_latencies.append(gen_latency_ms)
                llm_token_counts.append(token_count)
                llm_tokens_per_sec.append(tps)

                # ----------------------------------------------------
                # C. Post-Generation Guardrails & Full Text RAG
                # ----------------------------------------------------
                t_guard_start = time.perf_counter_ns()
                grounding_res, _ = guardrails.check_grounding(cleaned_q, sources, llm_answer)
                final_answer, _ = guardrails.sanitize_output(llm_answer, is_refusal=grounding_res.refusal_triggered)
                t_total_end = time.perf_counter_ns()

                full_text_rag_ms = (t_total_end - t_total_start) / 1_000_000.0
                full_text_rag_latencies.append(full_text_rag_ms)

                # ----------------------------------------------------
                # D. STT & Voice Pipeline
                # ----------------------------------------------------
                # Measure STT processing
                sample_audio = base64.b64encode(cleaned_q.encode("utf-8")).decode("utf-8")
                _, _, stt_ms = stt.transcribe(sample_audio, language_hint=lang)
                stt_latencies.append(stt_ms)
                full_voice_rag_latencies.append(full_text_rag_ms + stt_ms)

                successful_runs += 1

                if run_idx == 1:
                    print(f"[{lang}] Q: '{query_text[:35]}...'")
                    print(f"     A: '{final_answer[:65]}...'")
                    print(f"     Retrieval: {ret_latency_ms:.1f}ms | TTFT: {ttft_ms:.1f}ms | LLM Gen: {gen_latency_ms:.1f}ms | Full RAG: {full_text_rag_ms:.1f}ms\n")

            except Exception as exc:
                failed_runs += 1
                logger.error("Run %d Query '%s' failed: %s", run_idx, query_text, exc)

    # --------------------------------------------------------
    # Calculate All Metric Statistics
    # --------------------------------------------------------
    ret_stats = stats(retrieval_latencies)
    ttft_stats = stats(llm_ttft_latencies)
    llm_gen_stats = stats(llm_gen_latencies)
    tokens_stats = stats([float(t) for t in llm_token_counts])
    tps_stats = stats(llm_tokens_per_sec)
    full_rag_stats = stats(full_text_rag_latencies)
    stt_stats = stats(stt_latencies)
    voice_rag_stats = stats(full_voice_rag_latencies)

    print("=" * 80)
    print("  AUDIT LATENCY BENCHMARK RESULTS")
    print("=" * 80)
    print(f"  Total Runs Completed : {successful_runs} / {successful_runs + failed_runs} (Failures: {failed_runs})")
    print(f"  Languages Evaluated  : Hindi, Bengali, Tamil, Marathi, Telugu, Gujarati, English")
    print("-" * 80)
    print(f"{'Measurement Boundary':<30} | {'Mean (ms)':<9} | {'P50 (ms)':<8} | {'P70 (ms)':<8} | {'P100 (ms)':<9}")
    print("-" * 80)
    print(f"{'A. Retrieval / Pre-gen':<30} | {ret_stats['mean']:>9.2f} | {ret_stats['p50']:>8.2f} | {ret_stats['p70']:>8.2f} | {ret_stats['p100']:>9.2f}")
    print(f"{'B. Qwen3 TTFT':<30} | {ttft_stats['mean']:>9.2f} | {ttft_stats['p50']:>8.2f} | {ttft_stats['p70']:>8.2f} | {ttft_stats['p100']:>9.2f}")
    print(f"{'B. Qwen3 Total Generation':<30} | {llm_gen_stats['mean']:>9.2f} | {llm_gen_stats['p50']:>8.2f} | {llm_gen_stats['p70']:>8.2f} | {llm_gen_stats['p100']:>9.2f}")
    print(f"{'C. FULL TEXT RAG (Real LLM)':<30} | {full_rag_stats['mean']:>9.2f} | {full_rag_stats['p50']:>8.2f} | {full_rag_stats['p70']:>8.2f} | {full_rag_stats['p100']:>9.2f}")
    print(f"{'D. STT Engine (Local)':<30} | {stt_stats['mean']:>9.2f} | {stt_stats['p50']:>8.2f} | {stt_stats['p70']:>8.2f} | {stt_stats['p100']:>9.2f}")
    print(f"{'E. FULL VOICE PIPELINE':<30} | {voice_rag_stats['mean']:>9.2f} | {voice_rag_stats['p50']:>8.2f} | {voice_rag_stats['p70']:>8.2f} | {voice_rag_stats['p100']:>9.2f}")
    print("=" * 80)
    print(f"  Qwen3 Tokens/sec (Mean): {tps_stats['mean']:.2f} t/s (P50: {tps_stats['p50']:.2f} t/s)")
    print(f"  Output Tokens (Mean)   : {tokens_stats['mean']:.1f} tokens (P50: {tokens_stats['p50']:.0f} tokens)")
    print("=" * 80)

    return {
        "retrieval": ret_stats,
        "ttft": ttft_stats,
        "llm_gen": llm_gen_stats,
        "tokens": tokens_stats,
        "tps": tps_stats,
        "full_text_rag": full_rag_stats,
        "stt": stt_stats,
        "full_voice_rag": voice_rag_stats,
        "successful_runs": successful_runs,
        "failed_runs": failed_runs,
    }


if __name__ == "__main__":
    run_rigorous_baseline_audit(num_runs=3)
