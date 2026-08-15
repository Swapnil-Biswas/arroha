"""
evaluation/llm_baseline.py
--------------------------
Direct standalone benchmarking tool for LM Studio API (Qwen3 4B 2507 Q4_K_M).
Measures:
  - Non-streaming vs Streaming latency
  - Real TTFT (timestamp of first chunk delivery)
  - Pure generation time (time from first token to last token)
  - True token throughput (completion_tokens / pure_generation_time) vs end-to-end throughput
  - Simple Prompt vs Full RAG Context Prompt
  - Thinking Mode & Reasoning token inspection
  - Max token scaling (max_tokens = 8, 16, 32)
"""

from __future__ import annotations

import json
import logging
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
from openai import OpenAI

from app.generation.prompts import build_rag_prompt
from app.schemas.response import SourceDocument

LM_STUDIO_URL = "http://127.0.0.1:1234/v1"
MODEL_ID = "qwen/qwen3-4b-2507"


def calc_stats(values: list[float]) -> dict[str, float]:
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


def run_simple_prompt_benchmark(client: OpenAI, num_iterations: int = 10) -> dict[str, Any]:
    prompt = "Answer in one short sentence: What is the capital of India?"
    messages = [{"role": "user", "content": prompt}]

    # Warmup
    client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        max_tokens=16,
        temperature=0.1,
    )

    # 1. Non-Streaming Benchmark
    non_stream_totals = []
    non_stream_tokens = []
    non_stream_tps_e2e = []

    for _ in range(num_iterations):
        t0 = time.perf_counter_ns()
        res = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            max_tokens=16,
            temperature=0.1,
            stream=False,
        )
        t_total = (time.perf_counter_ns() - t0) / 1_000_000.0

        comp_tokens = res.usage.completion_tokens if res.usage else len(res.choices[0].message.content.split())
        prompt_tokens = res.usage.prompt_tokens if res.usage else 0

        non_stream_totals.append(t_total)
        non_stream_tokens.append(comp_tokens)
        non_stream_tps_e2e.append(comp_tokens / (t_total / 1000.0) if t_total > 0 else 0)

    # 2. Streaming Benchmark
    stream_ttfts = []
    stream_pure_gens = []  # Time from first token to end
    stream_totals = []
    stream_tokens = []
    stream_pure_tps = []   # Tokens / pure generation time
    stream_e2e_tps = []    # Tokens / total request time
    sample_content = ""

    for _ in range(num_iterations):
        t_start = time.perf_counter_ns()
        stream = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            max_tokens=16,
            temperature=0.1,
            stream=True,
        )

        t_first_token = None
        chunks = []
        chunk_count = 0

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                if t_first_token is None:
                    t_first_token = time.perf_counter_ns()
                chunks.append(chunk.choices[0].delta.content)
                chunk_count += 1

        t_end = time.perf_counter_ns()
        sample_content = "".join(chunks).strip()

        if t_first_token is None:
            t_first_token = t_end

        ttft_ms = (t_first_token - t_start) / 1_000_000.0
        pure_gen_ms = (t_end - t_first_token) / 1_000_000.0
        total_ms = (t_end - t_start) / 1_000_000.0

        # Each streamed chunk in llama.cpp corresponds to approximately 1 token
        actual_tokens = max(chunk_count, 1)

        stream_ttfts.append(ttft_ms)
        stream_pure_gens.append(pure_gen_ms)
        stream_totals.append(total_ms)
        stream_tokens.append(actual_tokens)

        pure_tps = actual_tokens / (pure_gen_ms / 1000.0) if pure_gen_ms > 0 else 0
        e2e_tps = actual_tokens / (total_ms / 1000.0) if total_ms > 0 else 0

        stream_pure_tps.append(pure_tps)
        stream_e2e_tps.append(e2e_tps)

    return {
        "prompt_tokens": prompt_tokens,
        "sample_output": sample_content,
        "non_streaming": {
            "total": calc_stats(non_stream_totals),
            "tokens": calc_stats([float(t) for t in non_stream_tokens]),
            "tps": calc_stats(non_stream_tps_e2e),
        },
        "streaming": {
            "ttft": calc_stats(stream_ttfts),
            "pure_gen": calc_stats(stream_pure_gens),
            "total": calc_stats(stream_totals),
            "tokens": calc_stats([float(t) for t in stream_tokens]),
            "pure_tps": calc_stats(stream_pure_tps),
            "e2e_tps": calc_stats(stream_e2e_tps),
        },
    }


def run_rag_prompt_benchmark(client: OpenAI, num_iterations: int = 10) -> dict[str, Any]:
    # Construct realistic RAG prompt
    sources = [
        SourceDocument(
            doc_id="1",
            text="नई दिल्ली भारत की राजधानी है और दिल्ली के राष्ट्रीय राजधानी क्षेत्र का हिस्सा है। इस शहर की आधारशिला 1911 में दिल्ली दरबार के दौरान सम्राट जॉर्ज पंचम द्वारा रखी गई थी।",
            language="hi",
            score=0.95,
        ),
        SourceDocument(
            doc_id="2",
            text="राष्ट्रपति भवन नई दिल्ली में राजपथ पर स्थित है और भारत के राष्ट्रपति का आधिकारिक आवास है।",
            language="hi",
            score=0.88,
        ),
    ]

    query = "भारत की राजधानी क्या है और इसका इतिहास क्या है?"
    sys_prompt, user_msg = build_rag_prompt(query, sources)
    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_msg}]

    # Measure prompt token count via non-streaming call
    res = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        max_tokens=30,
        temperature=0.1,
        stream=False,
    )
    prompt_tokens = res.usage.prompt_tokens if res.usage else 0

    stream_ttfts = []
    stream_pure_gens = []
    stream_totals = []
    stream_tokens = []
    stream_pure_tps = []
    stream_e2e_tps = []
    sample_content = ""

    for _ in range(num_iterations):
        t_start = time.perf_counter_ns()
        stream = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            max_tokens=30,
            temperature=0.1,
            stream=True,
        )

        t_first_token = None
        chunks = []
        chunk_count = 0

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                if t_first_token is None:
                    t_first_token = time.perf_counter_ns()
                chunks.append(chunk.choices[0].delta.content)
                chunk_count += 1

        t_end = time.perf_counter_ns()
        sample_content = "".join(chunks).strip()

        if t_first_token is None:
            t_first_token = t_end

        ttft_ms = (t_first_token - t_start) / 1_000_000.0
        pure_gen_ms = (t_end - t_first_token) / 1_000_000.0
        total_ms = (t_end - t_start) / 1_000_000.0
        actual_tokens = max(chunk_count, 1)

        stream_ttfts.append(ttft_ms)
        stream_pure_gens.append(pure_gen_ms)
        stream_totals.append(total_ms)
        stream_tokens.append(actual_tokens)

        pure_tps = actual_tokens / (pure_gen_ms / 1000.0) if pure_gen_ms > 0 else 0
        e2e_tps = actual_tokens / (total_ms / 1000.0) if total_ms > 0 else 0

        stream_pure_tps.append(pure_tps)
        stream_e2e_tps.append(e2e_tps)

    return {
        "prompt_tokens": prompt_tokens,
        "sample_output": sample_content,
        "ttft": calc_stats(stream_ttfts),
        "pure_gen": calc_stats(stream_pure_gens),
        "total": calc_stats(stream_totals),
        "tokens": calc_stats([float(t) for t in stream_tokens]),
        "pure_tps": calc_stats(stream_pure_tps),
        "e2e_tps": calc_stats(stream_e2e_tps),
    }


def run_max_tokens_scaling_benchmark(client: OpenAI, num_iterations: int = 5) -> dict[int, dict[str, Any]]:
    prompt = "Answer in one short sentence: What is the capital of India and why is it important?"
    messages = [{"role": "user", "content": prompt}]
    results = {}

    for max_tok in [8, 16, 32]:
        ttfts = []
        pure_gens = []
        totals = []
        tokens_list = []
        pure_tps_list = []

        for _ in range(num_iterations):
            t_start = time.perf_counter_ns()
            stream = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                max_tokens=max_tok,
                temperature=0.1,
                stream=True,
            )

            t_first = None
            chunk_count = 0
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    if t_first is None:
                        t_first = time.perf_counter_ns()
                    chunk_count += 1

            t_end = time.perf_counter_ns()
            if t_first is None:
                t_first = t_end

            ttft_ms = (t_first - t_start) / 1_000_000.0
            pure_gen_ms = (t_end - t_first) / 1_000_000.0
            total_ms = (t_end - t_start) / 1_000_000.0
            actual_toks = max(chunk_count, 1)

            ttfts.append(ttft_ms)
            pure_gens.append(pure_gen_ms)
            totals.append(total_ms)
            tokens_list.append(actual_toks)
            pure_tps_list.append(actual_toks / (pure_gen_ms / 1000.0) if pure_gen_ms > 0 else 0)

        results[max_tok] = {
            "ttft": calc_stats(ttfts),
            "pure_gen": calc_stats(pure_gens),
            "total": calc_stats(totals),
            "tokens": calc_stats([float(t) for t in tokens_list]),
            "pure_tps": calc_stats(pure_tps_list),
        }

    return results


def check_thinking_mode(client: OpenAI) -> dict[str, Any]:
    prompt = "Answer in one short sentence: What is the capital of India?"
    res = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0.1,
    )

    msg = res.choices[0].message
    content = msg.content or ""

    # Check for thinking fields or tags
    has_think_tags = "<think>" in content or "</think>" in content
    reasoning_field = getattr(msg, "reasoning_content", None)

    return {
        "has_think_tags": has_think_tags,
        "reasoning_field_present": reasoning_field is not None,
        "reasoning_content": reasoning_field,
        "full_content": content,
        "usage": {
            "prompt_tokens": res.usage.prompt_tokens if res.usage else 0,
            "completion_tokens": res.usage.completion_tokens if res.usage else 0,
            "total_tokens": res.usage.total_tokens if res.usage else 0,
        } if res.usage else {},
    }


def main():
    print("=" * 80)
    print("  HH GOA 2026: RIGOROUS LLM BASELINE & DISCREPANCY INVESTIGATION")
    print(f"  Target: {MODEL_ID} on {LM_STUDIO_URL}")
    print("=" * 80)

    client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio", timeout=30.0)

    # 1. Thinking Mode Check
    print("\n[1/4] Inspecting Qwen3 Thinking Mode & Raw Response...")
    think_info = check_thinking_mode(client)
    print(f"  Has <think> tags: {think_info['has_think_tags']}")
    print(f"  Reasoning Content Exposed: {think_info['reasoning_field_present']}")
    print(f"  Raw Content: '{think_info['full_content']}'")
    print(f"  Token Usage: {think_info['usage']}")

    # 2. Simple Prompt Benchmark (Non-streaming vs Streaming)
    print("\n[2/4] Benchmarking Simple Prompt (10 iterations)...")
    simple_res = run_simple_prompt_benchmark(client, num_iterations=10)
    print(f"  Simple Prompt Tokens: {simple_res['prompt_tokens']}")
    print(f"  Sample Output: '{simple_res['sample_output']}'")
    print(f"  Non-Streaming Total: P50 = {simple_res['non_streaming']['total']['p50']:.2f} ms (Mean = {simple_res['non_streaming']['total']['mean']:.2f} ms)")
    print(f"  Streaming TTFT:      P50 = {simple_res['streaming']['ttft']['p50']:.2f} ms (Mean = {simple_res['streaming']['ttft']['mean']:.2f} ms)")
    print(f"  Streaming Pure Gen:  P50 = {simple_res['streaming']['pure_gen']['p50']:.2f} ms (Mean = {simple_res['streaming']['pure_gen']['mean']:.2f} ms)")
    print(f"  Streaming Total:     P50 = {simple_res['streaming']['total']['p50']:.2f} ms (Mean = {simple_res['streaming']['total']['mean']:.2f} ms)")
    print(f"  Pure Generation Tok/s (P50): {simple_res['streaming']['pure_tps']['p50']:.2f} tok/s (Mean = {simple_res['streaming']['pure_tps']['mean']:.2f} tok/s)")
    print(f"  End-to-End Tok/s (P50):      {simple_res['streaming']['e2e_tps']['p50']:.2f} tok/s (Mean = {simple_res['streaming']['e2e_tps']['mean']:.2f} tok/s)")

    # 3. Actual RAG Prompt Benchmark
    print("\n[3/4] Benchmarking Full RAG Context Prompt (10 iterations)...")
    rag_res = run_rag_prompt_benchmark(client, num_iterations=10)
    print(f"  Full RAG Prompt Tokens: {rag_res['prompt_tokens']}")
    print(f"  Sample Output: '{rag_res['sample_output']}'")
    print(f"  Streaming TTFT:      P50 = {rag_res['ttft']['p50']:.2f} ms (Mean = {rag_res['ttft']['mean']:.2f} ms)")
    print(f"  Streaming Pure Gen:  P50 = {rag_res['pure_gen']['p50']:.2f} ms (Mean = {rag_res['pure_gen']['mean']:.2f} ms)")
    print(f"  Streaming Total:     P50 = {rag_res['total']['p50']:.2f} ms (Mean = {rag_res['total']['mean']:.2f} ms)")
    print(f"  Pure Generation Tok/s (P50): {rag_res['pure_tps']['p50']:.2f} tok/s (Mean = {rag_res['pure_tps']['mean']:.2f} tok/s)")
    print(f"  End-to-End Tok/s (P50):      {rag_res['e2e_tps']['p50']:.2f} tok/s (Mean = {rag_res['e2e_tps']['mean']:.2f} tok/s)")

    # 4. Max Tokens Scaling
    print("\n[4/4] Benchmarking Max Tokens Scaling (max_tokens = 8, 16, 32)...")
    scaling_res = run_max_tokens_scaling_benchmark(client, num_iterations=5)
    for max_t, data in scaling_res.items():
        print(f"  max_tokens={max_t:2d} | Tokens={data['tokens']['p50']:.0f} | TTFT P50={data['ttft']['p50']:6.2f} ms | Pure Gen P50={data['pure_gen']['p50']:6.2f} ms | Total P50={data['total']['p50']:6.2f} ms | Pure TPS={data['pure_tps']['p50']:5.2f} t/s")

    print("\n" + "=" * 80)
    print("  INVESTIGATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
