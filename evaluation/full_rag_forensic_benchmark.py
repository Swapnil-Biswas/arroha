"""
evaluation/full_rag_forensic_benchmark.py
-----------------------------------------
Authoritative End-to-End Full Text RAG Forensic Latency Benchmark.

Measures the exact production RAG path over the 50 canonical queries from benchmark.py
with stage-by-stage nanosecond instrumentation:
- Query Preprocessing / Input Guardrails
- Query Embedding (Optimized FP16 CUDA hot-path)
- FAISS Vector Retrieval (IndexFlatIP)
- BM25 Lexical Retrieval & Hybrid Fusion
- Prompt Assembly
- LLM Time-To-First-Token (TTFT)
- T3, T5, T10 Token Milestones
- Pure Generation / Decode Time
- Output Guardrails & Grounding Verification
- Total End-to-End Wall Clock Latency
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.config import (
    LATENCY_BUDGET_MS,
    LLM_ENDPOINT,
    LLM_MAX_TOKENS,
    LLM_MODEL_ID,
    LLM_TEMPERATURE,
    RETRIEVAL_TOP_K,
)
from app.generation.llm import LLMGenerator
from app.generation.prompts import build_rag_prompt
from app.guardrails.validator import GuardrailsValidator
from app.pipeline import RAGPipeline
from app.retrieval.hybrid import HybridRetriever
from app.schemas.query import QueryRequest
from benchmark import QUERIES, percentile

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

N_BENCHMARK_QUERIES = 50


class FullRAGForensicBenchmark:
    def __init__(self) -> None:
        logger.info("Initializing Full RAG Forensic Benchmark suite...")
        self.pipeline = RAGPipeline()
        self.raw_queries = [QUERIES[i % len(QUERIES)] for i in range(N_BENCHMARK_QUERIES)]
        logger.info("Loaded %d queries across %d unique prompt patterns.", len(self.raw_queries), len(QUERIES))

    def warmup(self) -> None:
        """Warm up retrieval models, FAISS, and llama-server."""
        logger.info("Warming up RAG pipeline and llama-server...")
        for _ in range(3):
            req = QueryRequest(query="What is FAISS used for?", top_k=RETRIEVAL_TOP_K)
            _ = self.pipeline.process_query(req)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        logger.info("Warmup complete.")

    def run_benchmark(self) -> dict[str, Any]:
        self.warmup()

        logger.info("Executing authoritative 50-query end-to-end RAG forensic benchmark...")

        stage_metrics = {
            "query_preprocessing_ms": [],
            "query_embedding_ms": [],
            "faiss_retrieval_ms": [],
            "bm25_retrieval_ms": [],
            "hybrid_fusion_ms": [],
            "total_retrieval_ms": [],
            "prompt_construction_ms": [],
            "llm_ttft_ms": [],
            "llm_t3_ms": [],
            "llm_t5_ms": [],
            "llm_decode_ms": [],
            "total_llm_ms": [],
            "grounding_and_output_ms": [],
            "total_e2e_rag_ms": [],
            "generated_tokens": [],
            "gen_throughput_tok_per_sec": [],
            "grounded_count": 0,
        }

        query_records = []

        for idx, query in enumerate(self.raw_queries, 1):
            t_req_start = time.perf_counter_ns()

            # 1. Query Preprocessing / Input Guardrail
            t_prep0 = time.perf_counter_ns()
            is_valid, cleaned_query, detected_script, error_reason, in_latency = self.pipeline.guardrails.validate_input(query)
            t_prep1 = time.perf_counter_ns()
            prep_ms = (t_prep1 - t_prep0) / 1e6

            # 2. Hybrid Retrieval with granular stages
            t_ret0 = time.perf_counter_ns()
            sources, ret_lat = self.pipeline.hybrid_retriever.search(cleaned_query, top_k=RETRIEVAL_TOP_K)
            t_ret1 = time.perf_counter_ns()
            total_ret_ms = (t_ret1 - t_ret0) / 1e6

            embed_ms = ret_lat.get("query_embed_ms", 0.0)
            faiss_ms = ret_lat.get("vector_retrieval_ms", 0.0)
            bm25_ms = ret_lat.get("bm25_retrieval_ms", 0.0)
            fusion_ms = ret_lat.get("hybrid_fusion_ms", 0.0)

            # 3. Prompt Construction
            t_prompt0 = time.perf_counter_ns()
            system_prompt, user_msg = build_rag_prompt(cleaned_query, sources)
            t_prompt1 = time.perf_counter_ns()
            prompt_ms = (t_prompt1 - t_prompt0) / 1e6

            # 4. LLM Generation with TTFT, T3, T5 milestones
            t_llm0 = time.perf_counter_ns()
            stream_resp = self.pipeline.llm_generator.client.chat.completions.create(
                model=self.pipeline.llm_generator.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=self.pipeline.llm_generator.max_tokens,
                temperature=self.pipeline.llm_generator.temperature,
                stream=True,
                stream_options={"include_usage": True},
            )

            t_first = None
            t_third = None
            t_fifth = None
            tokens = []
            usage_tokens = None

            for chunk in stream_resp:
                t_chunk = time.perf_counter_ns()
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_tokens = chunk.usage.completion_tokens
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    tok = chunk.choices[0].delta.content
                    tokens.append(tok)
                    tok_idx = len(tokens)
                    if t_first is None:
                        t_first = t_chunk
                    if tok_idx == 3 and t_third is None:
                        t_third = t_chunk
                    if tok_idx == 5 and t_fifth is None:
                        t_fifth = t_chunk

            t_llm_end = time.perf_counter_ns()

            ttft_ms = (t_first - t_llm0) / 1e6 if t_first else (t_llm_end - t_llm0) / 1e6
            t3_ms = (t_third - t_llm0) / 1e6 if t_third else ttft_ms
            t5_ms = (t_fifth - t_llm0) / 1e6 if t_fifth else t3_ms
            decode_ms = (t_llm_end - t_first) / 1e6 if t_first else 0.0
            total_llm_ms = (t_llm_end - t_llm0) / 1e6

            raw_answer = "".join(tokens).strip()
            final_token_count = usage_tokens if usage_tokens is not None else max(len(tokens), 1)
            gen_tps = final_token_count / (decode_ms / 1000.0) if decode_ms > 0 else 0.0

            # 5. Output Guardrails & Grounding Check
            t_post0 = time.perf_counter_ns()
            grounding_res, ground_ms = self.pipeline.guardrails.check_grounding(cleaned_query, sources, raw_answer)
            final_answer, _ = self.pipeline.guardrails.sanitize_output(raw_answer, is_refusal=grounding_res.refusal_triggered)
            t_post1 = time.perf_counter_ns()
            post_ms = (t_post1 - t_post0) / 1e6

            # 6. Total End-to-End Latency
            t_req_end = time.perf_counter_ns()
            total_e2e_ms = (t_req_end - t_req_start) / 1e6

            if grounding_res.is_grounded:
                stage_metrics["grounded_count"] += 1

            stage_metrics["query_preprocessing_ms"].append(prep_ms)
            stage_metrics["query_embedding_ms"].append(embed_ms)
            stage_metrics["faiss_retrieval_ms"].append(faiss_ms)
            stage_metrics["bm25_retrieval_ms"].append(bm25_ms)
            stage_metrics["hybrid_fusion_ms"].append(fusion_ms)
            stage_metrics["total_retrieval_ms"].append(total_ret_ms)
            stage_metrics["prompt_construction_ms"].append(prompt_ms)
            stage_metrics["llm_ttft_ms"].append(ttft_ms)
            stage_metrics["llm_t3_ms"].append(t3_ms)
            stage_metrics["llm_t5_ms"].append(t5_ms)
            stage_metrics["llm_decode_ms"].append(decode_ms)
            stage_metrics["total_llm_ms"].append(total_llm_ms)
            stage_metrics["grounding_and_output_ms"].append(post_ms)
            stage_metrics["total_e2e_rag_ms"].append(total_e2e_ms)
            stage_metrics["generated_tokens"].append(final_token_count)
            stage_metrics["gen_throughput_tok_per_sec"].append(gen_tps)

            query_records.append({
                "query_index": idx,
                "query": query,
                "answer": final_answer,
                "token_count": final_token_count,
                "embed_ms": round(embed_ms, 2),
                "faiss_ms": round(faiss_ms, 2),
                "bm25_ms": round(bm25_ms, 2),
                "retrieval_total_ms": round(total_ret_ms, 2),
                "ttft_ms": round(ttft_ms, 2),
                "t3_ms": round(t3_ms, 2),
                "t5_ms": round(t5_ms, 2),
                "decode_ms": round(decode_ms, 2),
                "total_llm_ms": round(total_llm_ms, 2),
                "total_e2e_ms": round(total_e2e_ms, 2),
                "is_grounded": grounding_res.is_grounded,
            })

            if idx % 10 == 0 or idx == N_BENCHMARK_QUERIES:
                logger.info(
                    "[%2d/50] Retrieval: %5.2fms | TTFT: %5.2fms | Decode: %6.2fms | Total RAG: %6.2fms",
                    idx, total_ret_ms, ttft_ms, decode_ms, total_e2e_ms
                )

        return self._summarize(stage_metrics, query_records)

    def _summarize(self, metrics: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
        stages = [
            ("query_preprocessing_ms", "Query Preprocessing / Input Guardrails"),
            ("query_embedding_ms", "Query Embedding (FP16 CUDA)"),
            ("faiss_retrieval_ms", "FAISS Search (IndexFlatIP)"),
            ("bm25_retrieval_ms", "SQLite BM25 Search"),
            ("hybrid_fusion_ms", "Hybrid Score Fusion"),
            ("total_retrieval_ms", "Total Retrieval Stage"),
            ("prompt_construction_ms", "Prompt Assembly"),
            ("llm_ttft_ms", "LLM Time-To-First-Token (TTFT)"),
            ("llm_t3_ms", "LLM T3 (3 Tokens Emitted)"),
            ("llm_t5_ms", "LLM T5 (5 Tokens Emitted)"),
            ("llm_decode_ms", "LLM Pure Token Generation / Decode"),
            ("total_llm_ms", "Total LLM Stage (TTFT + Decode)"),
            ("grounding_and_output_ms", "Output Sanitization & Grounding Check"),
            ("total_e2e_rag_ms", "TOTAL END-TO-END RAG PIPELINE"),
        ]

        summary_table = []
        p50_e2e = percentile(metrics["total_e2e_rag_ms"], 50)
        mean_e2e = statistics.mean(metrics["total_e2e_rag_ms"])

        for key, label in stages:
            vals = metrics[key]
            mean_val = statistics.mean(vals)
            p50_val = percentile(vals, 50)
            p95_val = percentile(vals, 95)
            p99_val = percentile(vals, 99)
            pct_of_total = (mean_val / mean_e2e) * 100.0 if mean_e2e > 0 else 0.0

            summary_table.append({
                "stage_key": key,
                "stage_name": label,
                "mean_ms": round(mean_val, 2),
                "p50_ms": round(p50_val, 2),
                "p95_ms": round(p95_val, 2),
                "p99_ms": round(p99_val, 2),
                "pct_of_total_time": round(pct_of_total, 2),
            })

        grounding_rate = (metrics["grounded_count"] / N_BENCHMARK_QUERIES) * 100.0
        avg_tokens = statistics.mean(metrics["generated_tokens"])
        mean_gen_tps = statistics.mean(metrics["gen_throughput_tok_per_sec"])

        return {
            "n_queries": N_BENCHMARK_QUERIES,
            "llm_model": LLM_MODEL_ID,
            "llm_endpoint": LLM_ENDPOINT,
            "max_tokens": LLM_MAX_TOKENS,
            "temperature": LLM_TEMPERATURE,
            "grounding_rate_pct": round(grounding_rate, 2),
            "avg_generated_tokens": round(avg_tokens, 2),
            "mean_generation_throughput_tok_per_sec": round(mean_gen_tps, 2),
            "stage_summary": summary_table,
            "query_records": records,
        }


def main():
    bench = FullRAGForensicBenchmark()
    res = bench.run_benchmark()

    # Print Formatted Table
    print("\n" + "=" * 105)
    print("  ARROHA END-TO-END FULL TEXT RAG FORENSIC LATENCY BENCHMARK (50 Queries)")
    print("=" * 105)
    print(f"{'Pipeline Stage':<42}{'Mean':>10}{'P50':>10}{'P95':>10}{'P99':>10}{'% of Total':>12}")
    print("-" * 105)

    for item in res["stage_summary"]:
        bold_tag = "**" if "TOTAL" in item["stage_name"] or "Total Retrieval" in item["stage_name"] or "Total LLM" in item["stage_name"] else ""
        print(
            f"{item['stage_name']:<42}"
            f"{item['mean_ms']:>9.2f}ms"
            f"{item['p50_ms']:>9.2f}ms"
            f"{item['p95_ms']:>9.2f}ms"
            f"{item['p99_ms']:>9.2f}ms"
            f"{item['pct_of_total_time']:>11.1f}%"
        )
    print("=" * 105)

    # Save to JSON
    json_path = BASE_DIR / "evaluation" / "results" / "full_rag_forensic_benchmark.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    logger.info("Saved telemetry to %s", json_path)

    # Save Markdown
    md_path = BASE_DIR / "evaluation" / "results" / "full_rag_forensic_benchmark.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# ARROHA End-to-End Full Text RAG Forensic Latency Benchmark\n\n")
        f.write("**LLM Candidate:** `Qwen2.5-1.5B-Instruct Q4_K_M` (`llama-server` CUDA b10451, `-ngl 99`, `--cache-reuse 64`)\n")
        f.write("**Embedding Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-d, FP16 CUDA hot-path)\n")
        f.write(f"**Retrieval Corpus:** 50,400 chunks in FAISS `IndexFlatIP` + SQLite FTS5\n")
        f.write(f"**Benchmark Dataset:** 50 canonical queries from `benchmark.py`\n\n")

        f.write("## 1. Stage-by-Stage Latency Breakdown (50 Queries)\n\n")
        f.write("| Pipeline Stage | Mean | P50 | P95 | P99 | Share of Total |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for item in res["stage_summary"]:
            f.write(
                f"| **{item['stage_name']}** | {item['mean_ms']:.2f} ms | **{item['p50_ms']:.2f} ms** | {item['p95_ms']:.2f} ms | {item['p99_ms']:.2f} ms | {item['pct_of_total_time']:.1f}% |\n"
            )

        f.write("\n## 2. Key Forensic Findings & Remaining Bottleneck\n\n")
        total_ret_item = [x for x in res["stage_summary"] if x["stage_key"] == "total_retrieval_ms"][0]
        total_llm_item = [x for x in res["stage_summary"] if x["stage_key"] == "total_llm_ms"][0]
        ttft_item = [x for x in res["stage_summary"] if x["stage_key"] == "llm_ttft_ms"][0]
        decode_item = [x for x in res["stage_summary"] if x["stage_key"] == "llm_decode_ms"][0]
        total_rag_item = [x for x in res["stage_summary"] if x["stage_key"] == "total_e2e_rag_ms"][0]

        f.write(f"- **Total Retrieval Latency:** **{total_ret_item['p50_ms']:.2f} ms P50** ({total_ret_item['pct_of_total_time']:.1f}% of total pipeline)\n")
        f.write(f"- **Total LLM Generation Latency:** **{total_llm_item['p50_ms']:.2f} ms P50** ({total_llm_item['pct_of_total_time']:.1f}% of total pipeline)\n")
        f.write(f"  - **TTFT (Prefill / Time to 1st Token):** **{ttft_item['p50_ms']:.2f} ms P50**\n")
        f.write(f"  - **Decode (Token Generation):** **{decode_item['p50_ms']:.2f} ms P50**\n")
        f.write(f"- **COMPLETE End-to-End Full RAG Latency:** **{total_rag_item['p50_ms']:.2f} ms P50** (P95: **{total_rag_item['p95_ms']:.2f} ms**)\n")
        f.write(f"- **Factual Grounding Accuracy:** **{res['grounding_rate_pct']:.1f}%**\n")

    logger.info("Saved report to %s", md_path)


if __name__ == "__main__":
    main()
