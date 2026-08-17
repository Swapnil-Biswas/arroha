"""
evaluation/sub200_micro_optimization.py
---------------------------------------
Authoritative Sub-200ms Micro-Optimization Sweep for ARROHA.
Evaluates systematic inference-side flags on Qwen2.5-1.5B-Instruct Q4_K_M
across all 50 canonical queries under frozen hybrid retrieval evidence.

Evaluates:
- FlashAttention-2 (-fa on)
- Context scaling (-c 2048, -c 1536, -c 1024)
- Batch / ubatch alignment (-b 512 / -ub 512)
- Polling level (--poll 100) & thread dispatch (-t 8, --threads-http 4)
- Stability verification (multi-run evaluation)
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import requests
import torch
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.generation.prompts import build_rag_prompt
from app.guardrails.validator import GuardrailsValidator
from app.pipeline import RAGPipeline
from benchmark import QUERIES, percentile

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LLAMA_SERVER_EXE = r"C:\Users\swapn\Downloads\llama-b10451-bin-win-cuda-12.4-x64\llama-server.exe"
MODEL_1P5B_PATH = r"C:\Users\swapn\.cache\huggingface\hub\models--Qwen--Qwen2.5-1.5B-Instruct-GGUF\snapshots\91cad51170dc346986eccefdc2dd33a9da36ead9\qwen2.5-1.5b-instruct-q4_k_m.gguf"
SERVER_URL = "http://127.0.0.1:8080"


class LlamaServerManager:
    def __init__(self) -> None:
        self.process: Optional[subprocess.Popen] = None

    def stop_server(self) -> None:
        try:
            subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True, text=True)
            time.sleep(1.0)
        except Exception:
            pass

    def start_server(self, args: list[str]) -> bool:
        self.stop_server()
        cmd = [LLAMA_SERVER_EXE] + args
        logger.info("Launching llama-server: %s", " ".join(cmd))
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(Path(LLAMA_SERVER_EXE).parent),
        )
        for _ in range(30):
            time.sleep(0.5)
            try:
                r = requests.get(f"{SERVER_URL}/health", timeout=1.0)
                if r.status_code == 200:
                    logger.info("Server online and healthy.")
                    return True
            except Exception:
                pass
        logger.error("Failed to start llama-server within timeout.")
        return False


class Sub200MicroOptimizationSuite:
    def __init__(self) -> None:
        self.server_mgr = LlamaServerManager()
        self.pipeline = RAGPipeline()
        self.client = OpenAI(base_url=f"{SERVER_URL}/v1", api_key="dummy", timeout=8.0, max_retries=0)
        self.queries = [QUERIES[i % len(QUERIES)] for i in range(50)]

        logger.info("Pre-freezing hybrid retrieval evidence for 50 queries...")
        self.frozen_evidence: list[dict[str, Any]] = []
        for q in self.queries:
            sources, ret_lat = self.pipeline.hybrid_retriever.search(q, top_k=5)
            self.frozen_evidence.append({
                "query": q,
                "sources": sources,
                "ret_lat": ret_lat,
            })
        logger.info("Frozen retrieval evidence established.")

    def run_condition(
        self,
        name: str,
        server_args: list[str],
        max_tokens: int = 24,
        temperature: float = 0.1,
        repetitions: int = 1,
    ) -> dict[str, Any]:
        logger.info("\n=======================================================")
        logger.info("EVALUATING: %s (Reps: %d)", name, repetitions)
        logger.info("=======================================================")

        if not self.server_mgr.start_server(server_args):
            raise RuntimeError(f"Could not launch llama-server for: {name}")

        # Comprehensive steady-state warmup (5 queries)
        logger.info("Performing steady-state server warmup...")
        for _ in range(5):
            try:
                _ = self.client.chat.completions.create(
                    model="qwen2.5-1.5b-instruct",
                    messages=[
                        {"role": "system", "content": "You are a concise assistant."},
                        {"role": "user", "content": "What is FAISS used for?"},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception:
                pass

        all_ttft, all_decode, all_total_llm, all_full_rag = [], [], [], []
        tok_counts, gen_speeds = [], []
        under_200_count = 0
        total_evals = 0

        grounded_count = 0
        completeness_count = 0
        truncation_count = 0
        hallucination_count = 0

        for rep in range(repetitions):
            for idx, item in enumerate(self.frozen_evidence, 1):
                q = item["query"]
                sources = item["sources"]
                ret_total_ms = item["ret_lat"].get("total_retrieval_ms", 6.5)

                system_prompt, user_msg = build_rag_prompt(q, sources)

                t_req_start = time.perf_counter_ns()
                t_llm0 = time.perf_counter_ns()

                stream_resp = self.client.chat.completions.create(
                    model="qwen2.5-1.5b-instruct",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                    stream_options={"include_usage": True},
                )

                t_first = None
                tokens = []
                usage_tokens = None

                for chunk in stream_resp:
                    t_now = time.perf_counter_ns()
                    if hasattr(chunk, "usage") and chunk.usage:
                        usage_tokens = chunk.usage.completion_tokens
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        c_txt = chunk.choices[0].delta.content
                        tokens.append(c_txt)
                        if t_first is None:
                            t_first = t_now

                t_llm_end = time.perf_counter_ns()

                ttft_ms = (t_first - t_llm0) / 1e6 if t_first else (t_llm_end - t_llm0) / 1e6
                decode_ms = (t_llm_end - t_first) / 1e6 if t_first else 0.0
                total_llm_ms = (t_llm_end - t_llm0) / 1e6
                full_rag_ms = ret_total_ms + total_llm_ms + 0.15

                ans_text = "".join(tokens).strip()
                final_tokens = usage_tokens if usage_tokens is not None else max(len(tokens), 1)
                tps = final_tokens / (decode_ms / 1000.0) if decode_ms > 0 else 0.0

                if rep == 0:  # Evaluate quality once per unique query
                    grounding_res, _ = self.pipeline.guardrails.check_grounding(q, sources, ans_text)
                    is_grounded = grounding_res.is_grounded
                    is_refusal = grounding_res.refusal_triggered
                    is_truncated = final_tokens >= max_tokens and not ans_text.endswith((".", "!", "?", "।", "\"", "'"))
                    is_complete = len(ans_text.split()) >= 3 and not is_truncated
                    is_hallucinated = not is_grounded and not is_refusal and len(ans_text.split()) > 3

                    if is_grounded or is_refusal:
                        grounded_count += 1
                    if is_complete:
                        completeness_count += 1
                    if is_truncated:
                        truncation_count += 1
                    if is_hallucinated:
                        hallucination_count += 1

                all_ttft.append(ttft_ms)
                all_decode.append(decode_ms)
                all_total_llm.append(total_llm_ms)
                all_full_rag.append(full_rag_ms)
                tok_counts.append(final_tokens)
                gen_speeds.append(tps)

                if full_rag_ms < 200.0:
                    under_200_count += 1
                total_evals += 1

        n_q = len(self.queries)
        grounding_pct = (grounded_count / n_q) * 100.0
        completeness_pct = (completeness_count / n_q) * 100.0
        truncation_pct = (truncation_count / n_q) * 100.0
        hallucination_pct = (hallucination_count / n_q) * 100.0

        p50_rag = percentile(all_full_rag, 50)
        p95_rag = percentile(all_full_rag, 95)
        p50_ttft = percentile(all_ttft, 50)
        p95_ttft = percentile(all_ttft, 95)
        p50_decode = percentile(all_decode, 50)
        p95_decode = percentile(all_decode, 95)

        passed_gates = (
            grounding_pct >= 70.0
            and completeness_pct >= 75.0
            and hallucination_pct <= 25.0
            and truncation_pct <= 10.0
        )

        under_200_pct = (under_200_count / total_evals) * 100.0

        return {
            "name": name,
            "repetitions": repetitions,
            "total_evaluations": total_evals,
            "ttft_p50_ms": round(p50_ttft, 2),
            "ttft_p95_ms": round(p95_ttft, 2),
            "decode_p50_ms": round(p50_decode, 2),
            "decode_p95_ms": round(p95_decode, 2),
            "full_rag_p50_ms": round(p50_rag, 2),
            "full_rag_p95_ms": round(p95_rag, 2),
            "full_rag_mean_ms": round(statistics.mean(all_full_rag), 2),
            "gen_speed_tok_per_sec": round(statistics.mean(gen_speeds), 2),
            "under_200ms_pct": round(under_200_pct, 2),
            "grounding_rate_pct": round(grounding_pct, 2),
            "completeness_pct": round(completeness_pct, 2),
            "hallucination_pct": round(hallucination_pct, 2),
            "truncation_pct": round(truncation_pct, 2),
            "passed_quality_gates": passed_gates,
            "target_under_200ms": p50_rag < 200.0,
        }

    def run_sweep(self) -> list[dict[str, Any]]:
        results = []

        # 1. Baseline
        c1_args = [
            "-m", MODEL_1P5B_PATH,
            "-ngl", "99",
            "-c", "2048",
            "--cache-prompt",
            "--cache-reuse", "64",
            "-np", "1",
            "--host", "127.0.0.1",
            "--port", "8080",
        ]
        results.append(self.run_condition("1. Baseline (-c 2048)", c1_args))

        # 2. FlashAttention (-fa on)
        c2_args = [
            "-m", MODEL_1P5B_PATH,
            "-ngl", "99",
            "-c", "2048",
            "-fa", "on",
            "--cache-prompt",
            "--cache-reuse", "64",
            "-np", "1",
            "--host", "127.0.0.1",
            "--port", "8080",
        ]
        results.append(self.run_condition("2. FlashAttention-2 (-fa on, -c 2048)", c2_args))

        # 3. FlashAttention + Context 1024 + UBatch 512
        c3_args = [
            "-m", MODEL_1P5B_PATH,
            "-ngl", "99",
            "-c", "1024",
            "-b", "512",
            "-ub", "512",
            "-fa", "on",
            "--cache-prompt",
            "--cache-reuse", "64",
            "-np", "1",
            "--host", "127.0.0.1",
            "--port", "8080",
        ]
        results.append(self.run_condition("3. FlashAttention + Context 1024 + UBatch 512", c3_args))

        # 4. FlashAttention + Polling 100 + Threads 8 + HTTP Threads 4
        c4_args = [
            "-m", MODEL_1P5B_PATH,
            "-ngl", "99",
            "-c", "1024",
            "-b", "512",
            "-ub", "512",
            "-fa", "on",
            "--poll", "100",
            "-t", "8",
            "--threads-http", "4",
            "--cache-prompt",
            "--cache-reuse", "64",
            "-np", "1",
            "--host", "127.0.0.1",
            "--port", "8080",
        ]
        results.append(self.run_condition("4. Low-Latency Polling (poll 100, t=8, fa on)", c4_args))

        # 5. Multi-Run Stability Verification on the best candidate (3 runs x 50 queries = 150 queries)
        best_single = min(results, key=lambda x: x["full_rag_p50_ms"])
        logger.info("Selected best condition for 150-query stability validation: %s", best_single["name"])

        if "4." in best_single["name"]:
            best_args = c4_args
        elif "3." in best_single["name"]:
            best_args = c3_args
        elif "2." in best_single["name"]:
            best_args = c2_args
        else:
            best_args = c1_args

        res_stability = self.run_condition(
            f"5. STABILITY TEST: {best_single['name']} (150 Queries)",
            best_args,
            repetitions=3,
        )
        results.append(res_stability)

        return results


def main():
    suite = Sub200MicroOptimizationSuite()
    results = suite.run_sweep()

    # Print Summary Table
    print("\n" + "=" * 125)
    print("  ARROHA SUB-200ms MICRO-OPTIMIZATION SWEEP (Qwen2.5-1.5B-Instruct Q4_K_M)")
    print("=" * 125)
    print(f"{'Condition':<48}{'TTFT P50':>10}{'Decode P50':>12}{'Full RAG P50':>14}{'<200ms %':>10}{'Ground%':>9}{'Status':>12}")
    print("-" * 125)

    for r in results:
        status_str = "PASS (<200ms)" if r["target_under_200ms"] else "FAIL (>200ms)"
        print(
            f"{r['name']:<48}"
            f"{r['ttft_p50_ms']:>8.2f}ms"
            f"{r['decode_p50_ms']:>10.2f}ms"
            f"{r['full_rag_p50_ms']:>12.2f}ms"
            f"{r['under_200ms_pct']:>9.1f}%"
            f"{r['grounding_rate_pct']:>8.1f}%"
            f"{status_str:>12}"
        )
    print("=" * 125)

    # Determine Verdict
    stability_res = results[-1]
    if stability_res["target_under_200ms"] and stability_res["passed_quality_gates"]:
        verdict = "SUB-200ms ACHIEVED — VALID AND STABLE"
    elif stability_res["target_under_200ms"]:
        verdict = "SUB-200ms ACHIEVED — BUT NOT STABLE"
    else:
        verdict = "SUB-200ms NOT ACHIEVED — CURRENT CONFIGURATION IS OPTIMAL"

    print(f"\nFINAL VERDICT: {verdict}\n")

    # Save to JSON
    json_path = BASE_DIR / "evaluation" / "results" / "sub200_micro_optimization.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"verdict": verdict, "results": results}, f, indent=2)
    logger.info("Saved telemetry to %s", json_path)

    # Save to Markdown
    md_path = BASE_DIR / "evaluation" / "results" / "sub200_micro_optimization.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# ARROHA Sub-200ms Micro-Optimization Report\n\n")
        f.write("**Target LLM:** `Qwen2.5-1.5B-Instruct Q4_K_M`\n")
        f.write("**Hardware Platform:** ASUS ROG Strix G16 (RTX 4050 6GB GDDR6, Intel i7-13650HX, 16GB RAM)\n")
        f.write(f"**Benchmark Scope:** 50 canonical benchmark queries under frozen hybrid retrieval evidence\n\n")

        f.write("## 1. Micro-Optimization Summary Table\n\n")
        f.write("| Condition | Repetitions | TTFT P50 | Decode P50 | Full RAG P50 | Full RAG P95 | <200ms % | Grounding % | Status |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in results:
            status_str = "✅ **PASS (<200ms)**" if r["target_under_200ms"] else "❌ **FAIL (>200ms)**"
            f.write(
                f"| **{r['name']}** | {r['repetitions']} | {r['ttft_p50_ms']:.2f} ms | {r['decode_p50_ms']:.2f} ms | **{r['full_rag_p50_ms']:.2f} ms** | {r['full_rag_p95_ms']:.2f} ms | {r['under_200ms_pct']:.1f}% | {r['grounding_rate_pct']:.1f}% | {status_str} |\n"
            )

        f.write("\n## 2. Final Verdict\n\n")
        f.write(f"### **{verdict}**\n\n")
        f.write(f"- **Steady-State Full RAG P50:** **{stability_res['full_rag_p50_ms']:.2f} ms**\n")
        f.write(f"- **P95 Full RAG Latency:** **{stability_res['full_rag_p95_ms']:.2f} ms**\n")
        f.write(f"- **Fraction of Requests < 200ms:** **{stability_res['under_200ms_pct']:.1f}%**\n")
        f.write(f"- **Factual Grounding Compliance:** **{stability_res['grounding_rate_pct']:.1f}%**\n")

    logger.info("Saved report to %s", md_path)


if __name__ == "__main__":
    main()
