"""
evaluation/benchmark_forensic_profiler.py
-----------------------------------------
High-resolution sub-stage profiler for the exact benchmark.py execution path.
"""

from __future__ import annotations

import json
import logging
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.retriever import get_retriever, warmup, search
from benchmark import QUERIES, percentile

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def profile_sub_stages(n: int = 50) -> dict[str, list[float]]:
    r = get_retriever()
    embedder = r.embedder
    tok = embedder.tokenizer
    model = embedder.torch_model
    device = embedder.device

    # Ensure device is synchronized and model warmed
    for _ in range(5):
        _ = embedder.embed_query("Warmup query")
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    timings = {
        "tokenizer_ms": [],
        "h2d_transfer_ms": [],
        "forward_pass_ms": [],
        "pooling_ms": [],
        "normalize_ms": [],
        "d2h_transfer_ms": [],
        "numpy_convert_ms": [],
        "total_embed_function_ms": [],
        "faiss_search_ms": [],
        "total_retrieval_ms": [],
    }

    for i in range(n):
        q = QUERIES[i % len(QUERIES)]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_total_start = time.perf_counter_ns()

        # 1. Tokenizer
        t0 = time.perf_counter_ns()
        encoded = tok(q, padding=True, truncation=True, max_length=128, return_tensors="pt")
        t1 = time.perf_counter_ns()
        timings["tokenizer_ms"].append((t1 - t0) / 1e6)

        # 2. H2D
        t0 = time.perf_counter_ns()
        input_ids = encoded["input_ids"].to(device, non_blocking=True)
        attn_mask = encoded["attention_mask"].to(device, non_blocking=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        timings["h2d_transfer_ms"].append((t1 - t0) / 1e6)

        # 3. Model Forward
        t0 = time.perf_counter_ns()
        with torch.inference_mode():
            out = model(input_ids=input_ids, attention_mask=attn_mask)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        timings["forward_pass_ms"].append((t1 - t0) / 1e6)

        # 4. Pooling
        t0 = time.perf_counter_ns()
        token_embeddings = out[0]
        if device == "cuda":
            input_mask_expanded = attn_mask.unsqueeze(-1).expand(token_embeddings.size()).half()
        else:
            input_mask_expanded = attn_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        timings["pooling_ms"].append((t1 - t0) / 1e6)

        # 5. Normalize
        t0 = time.perf_counter_ns()
        normed = torch.nn.functional.normalize(pooled, p=2, dim=1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        timings["normalize_ms"].append((t1 - t0) / 1e6)

        # 6. D2H + NumPy
        t0 = time.perf_counter_ns()
        vec_tensor = normed[0].float().cpu()
        t1 = time.perf_counter_ns()
        timings["d2h_transfer_ms"].append((t1 - t0) / 1e6)

        t0 = time.perf_counter_ns()
        vec = vec_tensor.numpy()
        t1 = time.perf_counter_ns()
        timings["numpy_convert_ms"].append((t1 - t0) / 1e6)

        t_embed_end = time.perf_counter_ns()
        timings["total_embed_function_ms"].append((t_embed_end - t_total_start) / 1e6)

        # 7. FAISS Search
        t0 = time.perf_counter_ns()
        results, s_lat = r.index_manager.search(vec, top_k=5)
        t1 = time.perf_counter_ns()
        timings["faiss_search_ms"].append((t1 - t0) / 1e6)
        timings["total_retrieval_ms"].append((t1 - t_total_start) / 1e6)

    return timings


def main():
    print("\n" + "=" * 80)
    print("  EXACT BENCHMARK.PY SUB-STAGE FORENSIC TIMING BREAKDOWN (50 Queries)")
    print("=" * 80)

    timings = profile_sub_stages(50)

    print(f"{'Stage / Sub-Stage':<35}{'Avg':>10}{'P50':>10}{'P95':>10}{'P99':>10}  (ms)")
    print("-" * 80)
    for name, vals in timings.items():
        print(
            f"{name:<35}"
            f"{statistics.mean(vals):>9.3f}ms"
            f"{percentile(vals, 50):>9.3f}ms"
            f"{percentile(vals, 95):>9.3f}ms"
            f"{percentile(vals, 99):>9.3f}ms"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
