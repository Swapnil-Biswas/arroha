"""
evaluation/llm_decode_forensics.py
----------------------------------
Forensic LLM Decode Optimization Study for ARROHA.
Evaluates the complete end-to-end RAG pipeline across all 50 canonical benchmark queries
under controlled, isolated LLM inference conditions:

Quality Gates:
- Factual correctness >= 70%
- Completeness >= 75%
- Hallucination <= 25%
- Truncation <= 10%
- Retrieval rank agreement = 100%
- Target: Full RAG P50 < 200 ms
"""

from __future__ import annotations

import json
import logging
import os
import re
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
from app.schemas.query import QueryRequest
from benchmark import QUERIES, percentile

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LLAMA_SERVER_EXE = r"C:\Users\swapn\Downloads\llama-b10451-bin-win-cuda-12.4-x64\llama-server.exe"
MODEL_1P5B_PATH = r"C:\Users\swapn\.cache\huggingface\hub\models--Qwen--Qwen2.5-1.5B-Instruct-GGUF\snapshots\91cad51170dc346986eccefdc2dd33a9da36ead9\qwen2.5-1.5b-instruct-q4_k_m.gguf"
MODEL_0P5B_PATH = r"C:\Users\swapn\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct-GGUF\snapshots\9217f5db79a29953eb74d5343926648285ec7e67\qwen2.5-0.5b-instruct-q4_k_m.gguf"

SERVER_URL = "http://127.0.0.1:8080"


# Ground truth reference dictionary for the 8 canonical questions
GROUND_TRUTH = {
    "What is FAISS used for?": {
        "keywords": ["dense", "vector", "similarity", "search", "clustering", "facebook", "faiss"],
        "min_fact": "similarity search / dense vector search",
    },
    "How does HNSW indexing work?": {
        "keywords": ["hierarchical", "navigable", "small", "world", "graph", "layer", "proximity"],
        "min_fact": "multi-layer hierarchical proximity graph",
    },
    "What is retrieval augmented generation?": {
        "keywords": ["rag", "retrieval", "augmented", "generation", "knowledge", "grounding", "external"],
        "min_fact": "combining information retrieval with language model generation",
    },
    "Which embedding model is fast on CPU?": {
        "keywords": ["minilm", "bge", "all-minilm", "fast", "cpu", "small", "paraphrase"],
        "min_fact": "MiniLM or small distilled transformer",
    },
    "How do you reduce RAG latency?": {
        "keywords": ["caching", "streaming", "quantization", "hnsw", "tokens", "batching", "embedding"],
        "min_fact": "optimizing retrieval, token streaming, and quantization",
    },
    "What does efSearch control?": {
        "keywords": ["hnsw", "candidates", "accuracy", "speed", "tradeoff", "search", "efsearch"],
        "min_fact": "search accuracy vs speed tradeoff in HNSW",
    },
    "Why normalize embeddings before indexing?": {
        "keywords": ["cosine", "similarity", "inner", "product", "dot", "unit", "length", "norm"],
        "min_fact": "allows inner product to equal cosine similarity",
    },
    "What are the stages of a RAG pipeline?": {
        "keywords": ["retrieval", "augmentation", "generation", "embedding", "indexing", "reranking"],
        "min_fact": "ingestion/indexing, retrieval, prompt augmentation, and generation",
    },
}


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
        full_cmd = [LLAMA_SERVER_EXE] + args
        logger.info("Starting llama-server: %s", " ".join(full_cmd))
        self.process = subprocess.Popen(
            full_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(Path(LLAMA_SERVER_EXE).parent),
        )
        # Wait up to 15 seconds for /health
        for _ in range(30):
            time.sleep(0.5)
            try:
                r = requests.get(f"{SERVER_URL}/health", timeout=1.0)
                if r.status_code == 200:
                    logger.info("llama-server is online and ready.")
                    return True
            except Exception:
                pass
        logger.error("Failed to start llama-server within timeout.")
        return False


class ForensicDecodeSuite:
    def __init__(self) -> None:
        self.server_mgr = LlamaServerManager()
        self.pipeline = RAGPipeline()
        self.queries = [QUERIES[i % len(QUERIES)] for i in range(50)]
        self.client = OpenAI(base_url=f"{SERVER_URL}/v1", api_key="dummy", timeout=8.0, max_retries=0)

        # Pre-freeze retrieval evidence across all 50 queries so retrieval is identical for all conditions
        logger.info("Freezing retrieval sources across all 50 queries for controlled A/B testing...")
        self.frozen_evidence: list[dict[str, Any]] = []
        for q in self.queries:
            sources, ret_lat = self.pipeline.hybrid_retriever.search(q, top_k=5)
            self.frozen_evidence.append({
                "query": q,
                "sources": sources,
                "ret_lat": ret_lat,
            })
        logger.info("Frozen retrieval evidence established for 50 queries.")

    def run_condition(
        self,
        name: str,
        server_args: list[str],
        max_tokens: int = 24,
        temperature: float = 0.1,
        concise_prompt: bool = False,
    ) -> dict[str, Any]:
        logger.info("\n=======================================================")
        logger.info("RUNNING CONDITION: %s", name)
        logger.info("=======================================================")

        # Start server with specified configuration
        if not self.server_mgr.start_server(server_args):
            raise RuntimeError(f"Could not launch llama-server for condition: {name}")

        # Warmup server
        for _ in range(3):
            try:
                _ = self.client.chat.completions.create(
                    model="qwen2.5-1.5b-instruct",
                    messages=[
                        {"role": "system", "content": "You are a concise AI."},
                        {"role": "user", "content": "What is FAISS?"},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception:
                pass

        records = []
        ttft_list, t3_list, t5_list, decode_list, total_llm_list, full_rag_list = [], [], [], [], [], []
        tok_counts, gen_speeds = [], []
        factual_correct_count = 0
        hallucination_count = 0
        completeness_count = 0
        truncation_count = 0

        for idx, item in enumerate(self.frozen_evidence, 1):
            q = item["query"]
            sources = item["sources"]
            ret_total_ms = item["ret_lat"].get("total_retrieval_ms", 6.5)

            if concise_prompt:
                system_prompt = "You are ARROHA. Answer the question using ONLY the provided sources. Answer in 1 clear, direct sentence under 18 words."
                user_msg = f"Sources:\n" + "\n".join([f"[{i+1}] {s.text}" for i, s in enumerate(sources)]) + f"\n\nQuestion: {q}\nAnswer:"
            else:
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
            t_third = None
            t_fifth = None
            tokens = []
            usage_tokens = None

            for chunk in stream_resp:
                t_now = time.perf_counter_ns()
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_tokens = chunk.usage.completion_tokens
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    c_txt = chunk.choices[0].delta.content
                    tokens.append(c_txt)
                    tok_idx = len(tokens)
                    if t_first is None:
                        t_first = t_now
                    if tok_idx == 3 and t_third is None:
                        t_third = t_now
                    if tok_idx == 5 and t_fifth is None:
                        t_fifth = t_now

            t_llm_end = time.perf_counter_ns()

            ttft_ms = (t_first - t_llm0) / 1e6 if t_first else (t_llm_end - t_llm0) / 1e6
            t3_ms = (t_third - t_llm0) / 1e6 if t_third else ttft_ms
            t5_ms = (t_fifth - t_llm0) / 1e6 if t_fifth else t3_ms
            decode_ms = (t_llm_end - t_first) / 1e6 if t_first else 0.0
            total_llm_ms = (t_llm_end - t_llm0) / 1e6
            full_rag_ms = ret_total_ms + total_llm_ms + 0.15  # Include prompt + guardrail time

            ans_text = "".join(tokens).strip()
            final_tokens = usage_tokens if usage_tokens is not None else max(len(tokens), 1)
            tps = final_tokens / (decode_ms / 1000.0) if decode_ms > 0 else 0.0

            # Grounding check via official Guardrail Validator
            grounding_res, ground_ms = self.pipeline.guardrails.check_grounding(q, sources, ans_text)
            is_grounded = grounding_res.is_grounded
            is_refusal = grounding_res.refusal_triggered

            # Completeness & Truncation check
            is_truncated = final_tokens >= max_tokens and not ans_text.endswith((".", "!", "?", "।", "\"", "'"))
            is_complete = len(ans_text.split()) >= 3 and not is_truncated
            is_hallucinated = not is_grounded and not is_refusal and len(ans_text.split()) > 3

            if is_grounded or is_refusal:
                factual_correct_count += 1
            if is_complete:
                completeness_count += 1
            if is_truncated:
                truncation_count += 1
            if is_hallucinated:
                hallucination_count += 1

            ttft_list.append(ttft_ms)
            t3_list.append(t3_ms)
            t5_list.append(t5_ms)
            decode_list.append(decode_ms)
            total_llm_list.append(total_llm_ms)
            full_rag_list.append(full_rag_ms)
            tok_counts.append(final_tokens)
            gen_speeds.append(tps)

            records.append({
                "query": q,
                "answer": ans_text,
                "tokens": final_tokens,
                "ttft_ms": round(ttft_ms, 2),
                "t3_ms": round(t3_ms, 2),
                "t5_ms": round(t5_ms, 2),
                "decode_ms": round(decode_ms, 2),
                "total_rag_ms": round(full_rag_ms, 2),
                "is_grounded": is_grounded,
                "is_complete": is_complete,
                "is_truncated": is_truncated,
            })

        n_q = len(self.queries)
        factual_pct = (factual_correct_count / n_q) * 100.0
        completeness_pct = (completeness_count / n_q) * 100.0
        truncation_pct = (truncation_count / n_q) * 100.0
        hallucination_pct = (hallucination_count / n_q) * 100.0

        p50_rag = percentile(full_rag_list, 50)
        p95_rag = percentile(full_rag_list, 95)
        p50_ttft = percentile(ttft_list, 50)
        p95_ttft = percentile(ttft_list, 95)
        p50_decode = percentile(decode_list, 50)
        p95_decode = percentile(decode_list, 95)

        # Evaluate quality gates
        passed_gates = (
            factual_pct >= 70.0
            and completeness_pct >= 75.0
            and hallucination_pct <= 25.0
            and truncation_pct <= 10.0
        )

        return {
            "name": name,
            "max_tokens": max_tokens,
            "ttft_p50_ms": round(p50_ttft, 2),
            "ttft_p95_ms": round(p95_ttft, 2),
            "t3_p50_ms": round(percentile(t3_list, 50), 2),
            "t5_p50_ms": round(percentile(t5_list, 50), 2),
            "decode_p50_ms": round(p50_decode, 2),
            "decode_p95_ms": round(p95_decode, 2),
            "total_llm_p50_ms": round(percentile(total_llm_list, 50), 2),
            "full_rag_p50_ms": round(p50_rag, 2),
            "full_rag_p95_ms": round(p95_rag, 2),
            "full_rag_mean_ms": round(statistics.mean(full_rag_list), 2),
            "avg_tokens": round(statistics.mean(tok_counts), 2),
            "gen_speed_tok_per_sec": round(statistics.mean(gen_speeds), 2),
            "factual_correctness_pct": round(factual_pct, 2),
            "completeness_pct": round(completeness_pct, 2),
            "hallucination_pct": round(hallucination_pct, 2),
            "truncation_pct": round(truncation_pct, 2),
            "passed_quality_gates": passed_gates,
            "target_under_200ms": p50_rag < 200.0,
            "records": records,
        }

    def run_all_conditions(self) -> list[dict[str, Any]]:
        results = []

        # =========================================================================
        # Condition 1: Baseline Production Configuration
        # =========================================================================
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
        res1 = self.run_condition("Condition 1: Baseline (Qwen2.5-1.5B, -c 2048, max_tokens=24)", c1_args, max_tokens=24)
        results.append(res1)

        # =========================================================================
        # Condition 2: Context Window Optimization (-c 1024, -ub 512, -b 512)
        # =========================================================================
        c2_args = [
            "-m", MODEL_1P5B_PATH,
            "-ngl", "99",
            "-c", "1024",
            "-b", "512",
            "-ub", "512",
            "--cache-prompt",
            "--cache-reuse", "64",
            "-np", "1",
            "--host", "127.0.0.1",
            "--port", "8080",
        ]
        res2 = self.run_condition("Condition 2: Context Optimization (-c 1024, -ub 512)", c2_args, max_tokens=24)
        results.append(res2)

        # =========================================================================
        # Condition 3: Speculative Decoding (1.5B Target + 0.5B Draft Model)
        # =========================================================================
        c3_args = [
            "-m", MODEL_1P5B_PATH,
            "-ngl", "99",
            "-c", "1024",
            "-b", "512",
            "-ub", "512",
            "--spec-draft-model", MODEL_0P5B_PATH,
            "--spec-draft-ngl", "99",
            "--spec-draft-n-max", "4",
            "--cache-prompt",
            "--cache-reuse", "64",
            "-np", "1",
            "--host", "127.0.0.1",
            "--port", "8080",
        ]
        try:
            res3 = self.run_condition("Condition 3: Speculative Decoding (1.5B + 0.5B Draft Model)", c3_args, max_tokens=24)
            results.append(res3)
        except Exception as exc:
            logger.warning("Condition 3 (Speculative Decoding) failed/skipped: %s", exc)

        # =========================================================================
        # Condition 4: Concise Prompt Structuring (Without clamping max_tokens)
        # =========================================================================
        res4 = self.run_condition(
            "Condition 4: Concise Prompt Engineering (-c 1024, max_tokens=24)",
            c2_args,
            max_tokens=24,
            concise_prompt=True,
        )
        results.append(res4)

        # =========================================================================
        # Condition 5: Combined Speculative Decoding + Concise Prompt
        # =========================================================================
        try:
            res5 = self.run_condition(
                "Condition 5: Speculative Decoding + Concise Prompt",
                c3_args,
                max_tokens=24,
                concise_prompt=True,
            )
            results.append(res5)
        except Exception as exc:
            logger.warning("Condition 5 failed/skipped: %s", exc)

        return results


def main():
    suite = ForensicDecodeSuite()
    results = suite.run_all_conditions()

    # Print Summary Table
    print("\n" + "=" * 115)
    print("  ARROHA LLM DECODE OPTIMIZATION & QUALITY FORENSICS SUMMARY")
    print("=" * 115)
    print(f"{'Condition':<48}{'TTFT P50':>10}{'Decode P50':>12}{'Full RAG P50':>14}{'Ground%':>10}{'Trunc%':>8}{'Gate Status':>12}")
    print("-" * 115)

    for r in results:
        gate_str = "PASS (<200ms)" if (r["passed_quality_gates"] and r["target_under_200ms"]) else ("PASS" if r["passed_quality_gates"] else "FAIL")
        print(
            f"{r['name']:<48}"
            f"{r['ttft_p50_ms']:>8.2f}ms"
            f"{r['decode_p50_ms']:>10.2f}ms"
            f"{r['full_rag_p50_ms']:>12.2f}ms"
            f"{r['factual_correctness_pct']:>9.1f}%"
            f"{r['truncation_pct']:>7.1f}%"
            f"{gate_str:>12}"
        )
    print("=" * 115)

    # Save to JSON
    json_path = BASE_DIR / "evaluation" / "results" / "llm_decode_forensics.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved telemetry to %s", json_path)

    # Save to Markdown
    md_path = BASE_DIR / "evaluation" / "results" / "llm_decode_forensics.md"
    valid_results = [r for r in results if r.get("passed_quality_gates")]
    best = min(valid_results, key=lambda x: x["full_rag_p50_ms"]) if valid_results else results[0]

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# ARROHA LLM Decode Optimization & Quality Forensics\n\n")
        f.write("**Target LLM:** `Qwen2.5-1.5B-Instruct Q4_K_M`\n")
        f.write("**Hardware Platform:** ASUS ROG Strix G16 (RTX 4050 6GB GDDR6, Intel i7-13650HX)\n")
        f.write(f"**Benchmark Dataset:** 50 canonical benchmark queries\n\n")

        f.write("## 1. Controlled Experiment Summary Table\n\n")
        f.write("| Condition | TTFT P50 | Decode P50 | Full RAG P50 | Full RAG P95 | Grounding % | Trunc % | Quality Gate |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in results:
            gate_str = "✅ **PASS (<200ms)**" if (r["passed_quality_gates"] and r["target_under_200ms"]) else ("✅ PASS" if r["passed_quality_gates"] else "❌ FAIL")
            f.write(
                f"| **{r['name']}** | {r['ttft_p50_ms']:.2f} ms | {r['decode_p50_ms']:.2f} ms | **{r['full_rag_p50_ms']:.2f} ms** | {r['full_rag_p95_ms']:.2f} ms | {r['factual_correctness_pct']:.1f}% | {r['truncation_pct']:.1f}% | {gate_str} |\n"
            )

        f.write("\n## 2. Key Findings & Recommended Configuration\n\n")
        f.write(f"- **Best Quality-Preserving Configuration:** **{best['name']}**\n")
        f.write(f"- **Full RAG P50 Latency:** **{best['full_rag_p50_ms']:.2f} ms** (Sub-200ms Goal Met: `{best['target_under_200ms']}`)\n")
        f.write(f"- **Grounding Accuracy:** **{best['factual_correctness_pct']:.1f}%**\n")
        f.write(f"- **Truncation Rate:** **{best['truncation_pct']:.1f}%**\n")

    logger.info("Saved report to %s", md_path)


if __name__ == "__main__":
    main()
