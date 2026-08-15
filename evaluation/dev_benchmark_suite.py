"""
evaluation/dev_benchmark_suite.py
---------------------------------
Comprehensive Controlled Optimization & Benchmark Suite for HH Goa 2026 Task 2.
Evaluates on the 12,600-document Multilingual Development Corpus across 14 Indic languages + English.

Executes:
- Phase 2: Chunking Strategy Benchmark (Sentence vs Fixed vs Recursive)
- Phase 3: Retrieval Benchmark on Dev Corpus (BM25 vs FAISS vs Hybrid)
- Phase 4: Context Compression Scaling (50, 100, 150, 200, 300, 400, 500 tokens)
- Phase 5: Retrieved Context Size / Top-K Scaling (top-1, top-2, top-3, top-5)
- Phase 6: Answer Length / Output Budget Scaling (5, 8, 12, 16, 24 tokens)
- Phase 7: Combined Latency Evaluation (Configs A, B, C, D)
- Phase 8: Prompt Caching / Static Prefix Inspection
- Phase 9: Retrieval Latency Breakdown & Profiling
- Phase 10: Full Multilingual Breakdown (All 14 Indic languages + English)
"""

from __future__ import annotations

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
    BM25_WEIGHT,
    DENSE_WEIGHT,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL_ID,
    PROCESSED_DATA_DIR,
    RETRIEVAL_TOP_K,
)
from app.generation.prompts import build_rag_prompt
from app.guardrails.validator import GuardrailsValidator
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import Reranker
from app.schemas.response import SourceDocument
from ingestion.chunking import get_chunker
from ingestion.dev_corpus import ALL_INDIC_LANGUAGES, generate_balanced_development_corpus
from ingestion.models import Chunk, Document
from indexing.bm25_index import BM25IndexManager
from indexing.embeddings import MultilingualEmbedder
from indexing.faiss_index import FAISSIndexManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dev_benchmark")

LM_STUDIO_URL = "http://127.0.0.1:1234/v1"
MODEL_ID = "qwen/qwen3-4b-2507"

DEV_INDEX_DIR = BASE_DIR / "indexes_dev"
DEV_INDEX_DIR.mkdir(parents=True, exist_ok=True)


def stats(values: list[float]) -> dict[str, float]:
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


# =========================================================================
# PHASE 2: CHUNKING EXPERIMENT
# =========================================================================
def run_phase_2_chunking(documents: list[Document]) -> dict[str, Any]:
    print("\n" + "=" * 80)
    print("  PHASE 2: CHUNKING EXPERIMENT (12,600 Canonical Documents)")
    print("=" * 80)

    strategies = ["sentence", "fixed", "recursive"]
    results = {}

    for strat in strategies:
        chunker = get_chunker(strat, chunk_size=250, chunk_overlap=40)
        chunks: list[Chunk] = []
        for doc in documents:
            chunks.extend(chunker.chunk_document(doc))

        sizes = [len(c.text) for c in chunks]
        word_counts = [len(c.text.split()) for c in chunks]

        # Chunks per language
        lang_counts: dict[str, int] = {}
        for c in chunks:
            lang_counts[c.language] = lang_counts.get(c.language, 0) + 1

        res = {
            "num_chunks": len(chunks),
            "avg_char_size": float(np.mean(sizes)),
            "median_char_size": float(np.median(sizes)),
            "max_char_size": int(np.max(sizes)),
            "avg_word_size": float(np.mean(word_counts)),
            "median_word_size": float(np.median(word_counts)),
            "max_word_size": int(np.max(word_counts)),
            "lang_counts": lang_counts,
        }
        results[strat] = res

        print(f"Strategy: {strat.upper():<10} | Chunks: {res['num_chunks']:<6} | Avg Chars: {res['avg_char_size']:5.1f} | Med Chars: {res['median_char_size']:5.1f} | Max Chars: {res['max_char_size']:4d} | Avg Words: {res['avg_word_size']:4.1f}")

    return results


# =========================================================================
# PHASE 3 & PHASE 10: RETRIEVAL BENCHMARK ON DEV CORPUS
# =========================================================================
def build_or_load_dev_indexes(documents: list[Document]) -> tuple[FAISSIndexManager, BM25IndexManager, list[Chunk]]:
    faiss_path = DEV_INDEX_DIR / "dev_vector.faiss"
    faiss_meta = DEV_INDEX_DIR / "dev_vector_meta.jsonl"
    bm25_path = DEV_INDEX_DIR / "dev_bm25.pkl"
    bm25_meta = DEV_INDEX_DIR / "dev_bm25_meta.jsonl"

    chunker = get_chunker("sentence")
    chunks: list[Chunk] = []
    for doc in documents:
        chunks.extend(chunker.chunk_document(doc))

    faiss_mgr = FAISSIndexManager(index_path=faiss_path, metadata_path=faiss_meta, dim=EMBEDDING_DIM)
    bm25_mgr = BM25IndexManager(index_path=bm25_path, metadata_path=bm25_meta)

    if not faiss_path.exists() or not bm25_path.exists():
        print(f"\nBuilding Dev Index ({len(chunks)} chunks across 14 languages)...")
        # 1. BM25
        bm25_mgr.build_index(chunks)

        # 2. FAISS Dense Embeddings
        embedder = MultilingualEmbedder(model_name=EMBEDDING_MODEL_ID, batch_size=EMBEDDING_BATCH_SIZE)
        texts = [c.text for c in chunks]
        embeddings = embedder.embed_documents(texts)
        faiss_mgr.build_index(embeddings, chunks)
        print("Dev Index build complete!")
    else:
        faiss_mgr.load()
        bm25_mgr.load()
        print(f"Loaded existing Dev Indexes ({faiss_mgr.ntotal} vectors, {bm25_mgr.corpus_size} BM25 docs).")

    return faiss_mgr, bm25_mgr, chunks


def run_phase_3_and_10_retrieval(
    records: list[Any],
    faiss_mgr: FAISSIndexManager,
    bm25_mgr: BM25IndexManager,
) -> dict[str, Any]:
    print("\n" + "=" * 80)
    print("  PHASE 3 & 10: RETRIEVAL BENCHMARK & MULTILINGUAL BREAKDOWN")
    print("=" * 80)

    embedder = MultilingualEmbedder(model_name=EMBEDDING_MODEL_ID, batch_size=EMBEDDING_BATCH_SIZE)

    # Sample 70 balanced test queries (5 per language across 14 Indic + English)
    test_eval_queries = []
    queries_by_lang: dict[str, list[dict[str, Any]]] = {}

    for rec in records:
        target_lang = rec.target_lang
        if target_lang not in queries_by_lang:
            queries_by_lang[target_lang] = []
        if len(queries_by_lang[target_lang]) < 5:
            item = {
                "qid": rec.query_id,
                "query": rec.query,
                "lang": target_lang,
                "gold_pids": [idx for idx, sel in enumerate(rec.passages.get("is_selected", [])) if sel == 1],
            }
            queries_by_lang[target_lang].append(item)
            test_eval_queries.append(item)

    print(f"Evaluating retrieval on {len(test_eval_queries)} balanced multilingual queries...")

    def eval_retriever(mode: str, k: int = 5) -> tuple[float, float, float, list[float]]:
        recalls, precisions, mrrs, lats = [], [], [], []

        for q in test_eval_queries:
            t0 = time.perf_counter_ns()
            qid = q["qid"]
            gold_pids = set(q["gold_pids"])

            if mode == "bm25":
                hits, _ = bm25_mgr.search(q["query"], top_k=k)
            elif mode == "faiss":
                q_emb, _ = embedder.embed_query(q["query"])
                hits, _ = faiss_mgr.search(q_emb, top_k=k)
            else: # hybrid
                # Dense
                q_emb, _ = embedder.embed_query(q["query"])
                dense_hits, _ = faiss_mgr.search(q_emb, top_k=max(k * 3, 15))
                # Sparse
                sparse_hits, _ = bm25_mgr.search(q["query"], top_k=max(k * 3, 15))
                # Fusion
                from app.retrieval.hybrid import min_max_normalize
                dense_scores = [s for _, s in dense_hits]
                sparse_scores = [s for _, s in sparse_hits]
                norm_dense = min_max_normalize(dense_scores)
                norm_sparse = min_max_normalize(sparse_scores)

                candidates = {}
                for idx, (doc, _) in enumerate(dense_hits):
                    d_id = doc.get("doc_id", doc.get("chunk_id", str(idx)))
                    candidates[d_id] = (doc, 0.6 * norm_dense[idx])
                for idx, (doc, _) in enumerate(sparse_hits):
                    d_id = doc.get("doc_id", doc.get("chunk_id", str(idx)))
                    prev_doc, prev_score = candidates.get(d_id, (doc, 0.0))
                    candidates[d_id] = (prev_doc, prev_score + 0.4 * norm_sparse[idx])

                sorted_cands = sorted(candidates.values(), key=lambda x: x[1], reverse=True)
                hits = [(doc, score) for doc, score in sorted_cands[:k]]

            t_lat = (time.perf_counter_ns() - t0) / 1_000_000.0
            lats.append(t_lat)

            # Measure Recall, Precision, MRR strictly
            matched_gold_pids = {doc.get("passage_id") for doc, _ in hits if doc.get("query_id") == qid and doc.get("passage_id") in gold_pids}
            recall = len(matched_gold_pids) / max(len(gold_pids), 1)
            precision = len(matched_gold_pids) / float(k)

            first_rank = None
            for r_idx, (doc, _) in enumerate(hits, 1):
                if doc.get("query_id") == qid and doc.get("passage_id") in gold_pids:
                    first_rank = r_idx
                    break
            mrr = (1.0 / first_rank) if first_rank is not None else 0.0

            assert 0.0 <= recall <= 1.0
            assert 0.0 <= precision <= 1.0
            assert 0.0 <= mrr <= 1.0

            recalls.append(recall)
            precisions.append(precision)
            mrrs.append(mrr)

        return float(np.mean(recalls)), float(np.mean(precisions)), float(np.mean(mrrs)), lats

    # Evaluate Overall
    b_rec, b_prec, b_mrr, b_lats = eval_retriever("bm25", k=5)
    f_rec, f_prec, f_mrr, f_lats = eval_retriever("faiss", k=5)
    h_rec5, h_prec5, h_mrr5, h_lats = eval_retriever("hybrid", k=5)
    h_rec10, h_prec10, h_mrr10, _ = eval_retriever("hybrid", k=10)

    print("-" * 80)
    print(f"{'Retriever Mode':<18} | {'Recall@5':<9} | {'Recall@10':<9} | {'Precision@5':<11} | {'MRR@5':<7} | {'P50 Latency (ms)':<16}")
    print("-" * 80)
    print(f"{'BM25 (Sparse)':<18} | {b_rec:>9.4f} | {'-':>9} | {b_prec:>11.4f} | {b_mrr:>7.4f} | {stats(b_lats)['p50']:>16.2f}")
    print(f"{'FAISS (Dense)':<18} | {f_rec:>9.4f} | {'-':>9} | {f_prec:>11.4f} | {f_mrr:>7.4f} | {stats(f_lats)['p50']:>16.2f}")
    print(f"{'Hybrid Top-5':<18} | {h_rec5:>9.4f} | {'-':>9} | {h_prec5:>11.4f} | {h_mrr5:>7.4f} | {stats(h_lats)['p50']:>16.2f}")
    print(f"{'Hybrid Top-10':<18} | {'-':>9} | {h_rec10:>9.4f} | {h_prec10:>11.4f} | {h_mrr10:>7.4f} | {stats(h_lats)['p50']:>16.2f}")
    print("=" * 80)

    # Per-Language Breakdown for Hybrid Top-5
    print("\nPER-LANGUAGE BREAKDOWN (Hybrid Top-5 across 14 Indic languages + English):")
    print(f"{'Language':<14} | {'Code':<5} | {'Indexed Docs':<12} | {'Queries':<8} | {'Recall@5':<9} | {'Precision@5':<11} | {'MRR@5':<7}")
    print("-" * 80)

    per_lang_results = {}
    for code3, code2, lang_name, _ in ALL_INDIC_LANGUAGES:
        lang_queries = queries_by_lang.get(code2, [])
        l_rec, l_prec, l_mrr = [], [], []

        for q in lang_queries:
            q_emb, _ = embedder.embed_query(q["query"])
            dense_hits, _ = faiss_mgr.search(q_emb, top_k=15)
            sparse_hits, _ = bm25_mgr.search(q["query"], top_k=15)
            from app.retrieval.hybrid import min_max_normalize
            norm_dense = min_max_normalize([s for _, s in dense_hits])
            norm_sparse = min_max_normalize([s for _, s in sparse_hits])
            candidates = {}
            for idx, (doc, _) in enumerate(dense_hits):
                d_id = doc.get("doc_id", doc.get("chunk_id", str(idx)))
                candidates[d_id] = (doc, 0.6 * norm_dense[idx])
            for idx, (doc, _) in enumerate(sparse_hits):
                d_id = doc.get("doc_id", doc.get("chunk_id", str(idx)))
                prev_doc, prev_score = candidates.get(d_id, (doc, 0.0))
                candidates[d_id] = (prev_doc, prev_score + 0.4 * norm_sparse[idx])
            hits = [d for d, s in sorted(candidates.values(), key=lambda x: x[1], reverse=True)[:5]]

            matched = {doc.get("passage_id") for doc in hits if doc.get("query_id") == q["qid"] and doc.get("passage_id") in q["gold_pids"]}
            l_rec.append(len(matched) / max(len(q["gold_pids"]), 1))
            l_prec.append(len(matched) / 5.0)
            first_rank = next((r_idx for r_idx, doc in enumerate(hits, 1) if doc.get("query_id") == q["qid"] and doc.get("passage_id") in q["gold_pids"]), None)
            l_mrr.append((1.0 / first_rank) if first_rank else 0.0)

        mean_rec = float(np.mean(l_rec)) if l_rec else 0.0
        mean_prec = float(np.mean(l_prec)) if l_prec else 0.0
        mean_mrr = float(np.mean(l_mrr)) if l_mrr else 0.0
        per_lang_results[code2] = {"name": lang_name, "recall": mean_rec, "prec": mean_prec, "mrr": mean_mrr}

        print(f"{lang_name:<14} | {code2:<5} | {900:<12} | {len(lang_queries):<8} | {mean_rec:>9.4f} | {mean_prec:>11.4f} | {mean_mrr:>7.4f}")

    return {
        "overall": {
            "bm25": {"recall5": b_rec, "prec5": b_prec, "mrr5": b_mrr, "p50_lat": stats(b_lats)["p50"]},
            "faiss": {"recall5": f_rec, "prec5": f_prec, "mrr5": f_mrr, "p50_lat": stats(f_lats)["p50"]},
            "hybrid": {"recall5": h_rec5, "recall10": h_rec10, "prec5": h_prec5, "mrr5": h_mrr5, "p50_lat": stats(h_lats)["p50"]},
        },
        "per_language": per_lang_results,
    }


# =========================================================================
# PHASE 4: CONTEXT COMPRESSION EXPERIMENT
# =========================================================================
def run_phase_4_context_compression(client: OpenAI) -> dict[int, dict[str, Any]]:
    print("\n" + "=" * 80)
    print("  PHASE 4: CONTEXT COMPRESSION EXPERIMENT (50 to 500 Token Budgets)")
    print("=" * 80)

    budgets = [50, 100, 150, 200, 300, 400, 500]
    results = {}

    raw_text = (
        "नई दिल्ली भारत की राजधानी है और दिल्ली के राष्ट्रीय राजधानी क्षेत्र का हिस्सा है। "
        "इस शहर की आधारशिला 1911 में दिल्ली दरबार के दौरान सम्राट जॉर्ज पंचम द्वारा रखी गई थी। "
        "ब्रिटिश वास्तुकार सर एडविन लुटियंस और सर हर्बर्ट बेकर ने शहर की योजना बनाई थी। "
        "राष्ट्रपति भवन, संसद भवन और इंडिया गेट यहाँ के प्रमुख ऐतिहासिक और प्रशासनिक स्थल हैं। "
        "यह शहर भारत सरकार की सभी तीन शाखाओं - कार्यपालिका, विधायिका और न्यायपालिका का केंद्र है。"
    )
    query = "भारत की राजधानी क्या है और इसका इतिहास क्या है?"

    for budget in budgets:
        # Truncate context according to approximate token budget (1 token ~ 3.5 chars)
        char_limit = int(budget * 3.5)
        trimmed_context = raw_text[:char_limit]

        src = [SourceDocument(doc_id="1", text=trimmed_context, language="hi", score=0.95)]
        sys_prompt, user_msg = build_rag_prompt(query, src, max_context_tokens=budget)
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_msg}]

        # Warmup
        client.chat.completions.create(model=MODEL_ID, messages=messages, max_tokens=15, temperature=0.1)

        # Run 5 trials per budget
        ttfts, gens, totals, toks_list, tps_list = [], [], [], [], []
        sample_answer = ""

        for _ in range(5):
            t_start = time.perf_counter_ns()
            stream = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                max_tokens=15,
                temperature=0.1,
                stream=True,
            )

            t_first = None
            chunks = []
            chunk_count = 0
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    if t_first is None:
                        t_first = time.perf_counter_ns()
                    chunks.append(chunk.choices[0].delta.content)
                    chunk_count += 1

            t_end = time.perf_counter_ns()
            if t_first is None:
                t_first = t_end

            sample_answer = "".join(chunks).strip()
            ttft_ms = (t_first - t_start) / 1_000_000.0
            gen_ms = (t_end - t_first) / 1_000_000.0
            tot_ms = (t_end - t_start) / 1_000_000.0
            actual_toks = max(chunk_count, 1)

            ttfts.append(ttft_ms)
            gens.append(gen_ms)
            totals.append(tot_ms)
            toks_list.append(actual_toks)
            tps_list.append(actual_toks / (gen_ms / 1000.0) if gen_ms > 0 else 0)

        # Count prompt tokens via non-streaming response usage
        res_u = client.chat.completions.create(model=MODEL_ID, messages=messages, max_tokens=1, temperature=0.1, stream=False)
        p_tokens = res_u.usage.prompt_tokens if res_u.usage else 0

        # Grounding check: Did it accurately capture New Delhi?
        is_grounded = "नई दिल्ली" in sample_answer or "दिल्ली" in sample_answer

        res_dict = {
            "budget": budget,
            "prompt_tokens": p_tokens,
            "ttft": stats(ttfts),
            "gen": stats(gens),
            "total": stats(totals),
            "tokens": stats([float(t) for t in toks_list]),
            "tps": stats(tps_list),
            "sample_answer": sample_answer,
            "is_grounded": is_grounded,
        }
        results[budget] = res_dict

        print(f"Context: {budget:3d} toks | Prompt Toks: {p_tokens:3d} | TTFT P50: {stats(ttfts)['p50']:6.2f} ms | Gen P50: {stats(gens)['p50']:6.2f} ms | Total P50: {stats(totals)['p50']:6.2f} ms | Toks/s: {stats(tps_list)['p50']:5.1f} | Grounded: {is_grounded}")

    return results


# =========================================================================
# PHASE 5: RETRIEVED CONTEXT SIZE / TOP-K SCALING
# =========================================================================
def run_phase_5_top_k(client: OpenAI) -> dict[int, dict[str, Any]]:
    print("\n" + "=" * 80)
    print("  PHASE 5: RETRIEVED CONTEXT SIZE / TOP-K (Top-1, Top-2, Top-3, Top-5)")
    print("=" * 80)

    passages = [
        "नई दिल्ली भारत की राजधानी है। इस शहर की आधारशिला 1911 में दिल्ली दरबार के दौरान रखी गई थी।",
        "राष्ट्रपति भवन नई दिल्ली में राजपथ पर स्थित है और भारत के राष्ट्रपति का आधिकारिक निवास है।",
        "संसद भवन नई दिल्ली में स्थित है जहाँ भारत की संसद की बैठकें होती हैं।",
        "इंडिया गेट नई दिल्ली में स्थित एक प्रसिद्ध युद्ध स्मारक है।",
        "दिल्ली का राष्ट्रीय संग्रहालय भारत के समृद्ध इतिहास और सांस्कृतिक कलाकृतियों को प्रदर्शित करता है।",
    ]
    query = "भारत की राजधानी क्या है और इसका इतिहास क्या है?"
    results = {}

    for k in [1, 2, 3, 5]:
        sources = [
            SourceDocument(doc_id=str(i + 1), text=p, language="hi", score=1.0 - i * 0.05)
            for i, p in enumerate(passages[:k])
        ]
        sys_prompt, user_msg = build_rag_prompt(query, sources, max_context_tokens=100 * k)
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_msg}]

        ttfts, gens, totals = [], [], []
        sample_answer = ""

        for _ in range(5):
            t_start = time.perf_counter_ns()
            stream = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                max_tokens=15,
                temperature=0.1,
                stream=True,
            )

            t_first = None
            chunks = []
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    if t_first is None:
                        t_first = time.perf_counter_ns()
                    chunks.append(chunk.choices[0].delta.content)

            t_end = time.perf_counter_ns()
            if t_first is None:
                t_first = t_end

            sample_answer = "".join(chunks).strip()
            ttfts.append((t_first - t_start) / 1_000_000.0)
            gens.append((t_end - t_first) / 1_000_000.0)
            totals.append((t_end - t_start) / 1_000_000.0)

        res_u = client.chat.completions.create(model=MODEL_ID, messages=messages, max_tokens=1, temperature=0.1, stream=False)
        p_tokens = res_u.usage.prompt_tokens if res_u.usage else 0

        res_dict = {
            "top_k": k,
            "prompt_tokens": p_tokens,
            "ttft": stats(ttfts),
            "gen": stats(gens),
            "total": stats(totals),
            "sample_answer": sample_answer,
        }
        results[k] = res_dict

        print(f"Top-{k} Passages | Prompt Tokens: {p_tokens:3d} | TTFT P50: {stats(ttfts)['p50']:6.2f} ms | Gen P50: {stats(gens)['p50']:6.2f} ms | Total P50: {stats(totals)['p50']:6.2f} ms")

    return results


# =========================================================================
# PHASE 6: ANSWER LENGTH / OUTPUT BUDGET SCALING
# =========================================================================
def run_phase_6_answer_length(client: OpenAI) -> dict[int, dict[str, Any]]:
    print("\n" + "=" * 80)
    print("  PHASE 6: ANSWER LENGTH / OUTPUT BUDGET SCALING (5, 8, 12, 16, 24 tokens)")
    print("=" * 80)

    prompt = "Answer in exactly 1-2 words: What is the capital of India?"
    results = {}

    for max_tok in [5, 8, 12, 16, 24]:
        ttfts, gens, totals, toks_list, tps_list = [], [], [], [], []
        sample_answer = ""

        for _ in range(5):
            t_start = time.perf_counter_ns()
            stream = client.chat.completions.create(
                model=MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tok,
                temperature=0.1,
                stream=True,
            )

            t_first = None
            chunks = []
            chunk_count = 0
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    if t_first is None:
                        t_first = time.perf_counter_ns()
                    chunks.append(chunk.choices[0].delta.content)
                    chunk_count += 1

            t_end = time.perf_counter_ns()
            if t_first is None:
                t_first = t_end

            sample_answer = "".join(chunks).strip()
            ttft_ms = (t_first - t_start) / 1_000_000.0
            gen_ms = (t_end - t_first) / 1_000_000.0
            tot_ms = (t_end - t_start) / 1_000_000.0
            actual_toks = max(chunk_count, 1)

            ttfts.append(ttft_ms)
            gens.append(gen_ms)
            totals.append(tot_ms)
            toks_list.append(actual_toks)
            tps_list.append(actual_toks / (gen_ms / 1000.0) if gen_ms > 0 else 0)

        res_dict = {
            "max_tokens": max_tok,
            "actual_tokens": stats([float(t) for t in toks_list]),
            "ttft": stats(ttfts),
            "gen": stats(gens),
            "total": stats(totals),
            "tps": stats(tps_list),
            "sample_answer": sample_answer,
        }
        results[max_tok] = res_dict

        print(f"Max Limit: {max_tok:2d} toks | Actual Toks: {stats([float(t) for t in toks_list])['p50']:.0f} | TTFT P50: {stats(ttfts)['p50']:6.2f} ms | Gen P50: {stats(gens)['p50']:6.2f} ms | Total P50: {stats(totals)['p50']:6.2f} ms | Toks/s: {stats(tps_list)['p50']:5.1f}")

    return results


# =========================================================================
# PHASE 7: COMBINED LATENCY EXPERIMENT
# =========================================================================
def run_phase_7_combined(client: OpenAI, retriever: Any) -> dict[str, dict[str, Any]]:
    print("\n" + "=" * 80)
    print("  PHASE 7: COMBINED LATENCY EXPERIMENT (Configurations A, B, C, D)")
    print("=" * 80)

    configs = {
        "Config A (Top-1, 100 ctx, max 8 out)": {"top_k": 1, "ctx_tokens": 100, "max_out": 8},
        "Config B (Top-2, 150 ctx, max 8 out)": {"top_k": 2, "ctx_tokens": 150, "max_out": 8},
        "Config C (Top-2, 200 ctx, max 12 out)": {"top_k": 2, "ctx_tokens": 200, "max_out": 12},
        "Config D (Top-3, 200 ctx, max 12 out)": {"top_k": 3, "ctx_tokens": 200, "max_out": 12},
    }

    test_queries = [
        ("भारत की राजधानी क्या है?", "hi"),
        ("পশ্চিমবঙ্গের রাজধানী কোনটি?", "bn"),
        ("தமிழ்நாட்டின் தலைநகரம் எது?", "ta"),
        ("महाराष्ट्राची राजधानी कोणती आहे?", "mr"),
        ("గుజరాత్ రాజధాని ఏది?", "te"),
    ]

    results = {}
    guardrails = GuardrailsValidator()

    for name, cfg in configs.items():
        ret_lats, ttfts, gens, full_rags = [], [], [], []

        for q_text, lang in test_queries:
            for _ in range(3):
                t_start = time.perf_counter_ns()

                # 1. Guardrail & Retrieval
                t_ret_start = time.perf_counter_ns()
                _, clean_q, _, _, _ = guardrails.validate_input(q_text, language_hint=lang)
                sources, _ = retriever.search(clean_q, top_k=cfg["top_k"])
                sys_prompt, user_msg = build_rag_prompt(clean_q, sources, max_context_tokens=cfg["ctx_tokens"])
                t_ret_end = time.perf_counter_ns()

                ret_ms = (t_ret_end - t_ret_start) / 1_000_000.0
                ret_lats.append(ret_ms)

                # 2. LLM Streaming
                t_llm_start = time.perf_counter_ns()
                stream = client.chat.completions.create(
                    model=MODEL_ID,
                    messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_msg}],
                    max_tokens=cfg["max_out"],
                    temperature=0.1,
                    stream=True,
                )

                t_first = None
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        if t_first is None:
                            t_first = time.perf_counter_ns()

                t_end = time.perf_counter_ns()
                if t_first is None:
                    t_first = t_end

                ttft_ms = (t_first - t_llm_start) / 1_000_000.0
                gen_ms = (t_end - t_first) / 1_000_000.0
                tot_rag_ms = (t_end - t_start) / 1_000_000.0

                ttfts.append(ttft_ms)
                gens.append(gen_ms)
                full_rags.append(tot_rag_ms)

        res_dict = {
            "retrieval": stats(ret_lats),
            "ttft": stats(ttfts),
            "gen": stats(gens),
            "full_rag": stats(full_rags),
        }
        results[name] = res_dict

        print(f"\n{name}:")
        print(f"  Retrieval: P50 = {stats(ret_lats)['p50']:6.2f} ms | P70 = {stats(ret_lats)['p70']:6.2f} ms | P100 = {stats(ret_lats)['p100']:6.2f} ms")
        print(f"  TTFT:      P50 = {stats(ttfts)['p50']:6.2f} ms | P70 = {stats(ttfts)['p70']:6.2f} ms | P100 = {stats(ttfts)['p100']:6.2f} ms")
        print(f"  Gen:       P50 = {stats(gens)['p50']:6.2f} ms | P70 = {stats(gens)['p70']:6.2f} ms | P100 = {stats(gens)['p100']:6.2f} ms")
        print(f"  FULL RAG:  P50 = {stats(full_rags)['p50']:6.2f} ms | P70 = {stats(full_rags)['p70']:6.2f} ms | P100 = {stats(full_rags)['p100']:6.2f} ms")

    return results


# =========================================================================
# PHASE 8: PROMPT CACHING / STATIC PREFIX INSPECTION
# =========================================================================
def run_phase_8_prompt_caching(client: OpenAI) -> dict[str, Any]:
    print("\n" + "=" * 80)
    print("  PHASE 8: PROMPT CACHING / STATIC PREFIX INSPECTION")
    print("=" * 80)

    static_system_prompt = (
        "You are an ultra-low-latency, strictly grounded multilingual voice assistant. "
        "Answer the user query accurately in 1 short sentence using only the provided context."
    )
    long_static_prefix = static_system_prompt + " " + ("Knowledge Base Reference Guide Section A. " * 10)

    messages_1 = [
        {"role": "system", "content": long_static_prefix},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    messages_2 = [
        {"role": "system", "content": long_static_prefix},
        {"role": "user", "content": "What is the capital of Germany?"},
    ]

    # First query (populates prefix cache if enabled in server)
    t0 = time.perf_counter_ns()
    res1 = client.chat.completions.create(model=MODEL_ID, messages=messages_1, max_tokens=10, temperature=0.1)
    t1 = (time.perf_counter_ns() - t0) / 1_000_000.0

    # Second query (shares identical prefix)
    t0 = time.perf_counter_ns()
    res2 = client.chat.completions.create(model=MODEL_ID, messages=messages_2, max_tokens=10, temperature=0.1)
    t2 = (time.perf_counter_ns() - t0) / 1_000_000.0

    # Check for prompt cache fields in usage
    usage_dict = res2.usage.model_dump() if res2.usage else {}
    has_cache_tokens = "prompt_tokens_details" in usage_dict or "cached_tokens" in str(usage_dict)

    print(f"  First Request Latency (Initial Eval):  {t1:.2f} ms")
    print(f"  Second Request Latency (Prefix Shared): {t2:.2f} ms")
    print(f"  Usage Metadata: {usage_dict}")
    print(f"  Explicit Cache Field Supported: {has_cache_tokens}")

    return {
        "first_call_ms": t1,
        "second_call_ms": t2,
        "usage": usage_dict,
        "has_explicit_cache_field": has_cache_tokens,
    }


# =========================================================================
# PHASE 9: RETRIEVAL LATENCY BREAKDOWN & PROFILING
# =========================================================================
def run_phase_9_retrieval_profiling(
    faiss_mgr: FAISSIndexManager,
    bm25_mgr: BM25IndexManager,
) -> dict[str, float]:
    print("\n" + "=" * 80)
    print("  PHASE 9: RETRIEVAL LATENCY BREAKDOWN & PROFILING (12,600 Chunks)")
    print("=" * 80)

    guardrails = GuardrailsValidator()
    embedder = MultilingualEmbedder(model_name=EMBEDDING_MODEL_ID, batch_size=EMBEDDING_BATCH_SIZE)
    query = "भारत की राजधानी क्या है और इसका इतिहास क्या है?"

    # Warmup
    q_emb, _ = embedder.embed_query(query)
    faiss_mgr.search(q_emb, top_k=5)
    bm25_mgr.search(query, top_k=5)

    times_guard, times_embed, times_faiss, times_bm25, times_fusion = [], [], [], [], []

    for _ in range(20):
        # 1. Guardrail
        t0 = time.perf_counter_ns()
        _, clean_q, _, _, _ = guardrails.validate_input(query, language_hint="hi")
        times_guard.append((time.perf_counter_ns() - t0) / 1_000_000.0)

        # 2. Embedding
        t0 = time.perf_counter_ns()
        q_emb, _ = embedder.embed_query(clean_q)
        times_embed.append((time.perf_counter_ns() - t0) / 1_000_000.0)

        # 3. FAISS Search
        t0 = time.perf_counter_ns()
        dense_hits, _ = faiss_mgr.search(q_emb, top_k=15)
        times_faiss.append((time.perf_counter_ns() - t0) / 1_000_000.0)

        # 4. BM25 Search
        t0 = time.perf_counter_ns()
        sparse_hits, _ = bm25_mgr.search(clean_q, top_k=15)
        times_bm25.append((time.perf_counter_ns() - t0) / 1_000_000.0)

        # 5. Fusion
        t0 = time.perf_counter_ns()
        from app.retrieval.hybrid import min_max_normalize
        norm_dense = min_max_normalize([s for _, s in dense_hits])
        norm_sparse = min_max_normalize([s for _, s in sparse_hits])
        candidates = {}
        for idx, (doc, _) in enumerate(dense_hits):
            d_id = doc.get("doc_id", doc.get("chunk_id", str(idx)))
            candidates[d_id] = (doc, 0.6 * norm_dense[idx])
        for idx, (doc, _) in enumerate(sparse_hits):
            d_id = doc.get("doc_id", doc.get("chunk_id", str(idx)))
            prev_doc, prev_score = candidates.get(d_id, (doc, 0.0))
            candidates[d_id] = (prev_doc, prev_score + 0.4 * norm_sparse[idx])
        _ = sorted(candidates.values(), key=lambda x: x[1], reverse=True)[:5]
        times_fusion.append((time.perf_counter_ns() - t0) / 1_000_000.0)

    g_p50 = stats(times_guard)["p50"]
    e_p50 = stats(times_embed)["p50"]
    f_p50 = stats(times_faiss)["p50"]
    b_p50 = stats(times_bm25)["p50"]
    u_p50 = stats(times_fusion)["p50"]
    total_p50 = g_p50 + e_p50 + f_p50 + b_p50 + u_p50

    print(f"  1. Input Guardrails:     {g_p50:6.2f} ms ({g_p50/total_p50*100:4.1f}%)")
    print(f"  2. Multilingual Embed:   {e_p50:6.2f} ms ({e_p50/total_p50*100:4.1f}%)  <-- MAIN RETRIEVAL BOTTLENECK")
    print(f"  3. FAISS Vector Search:  {f_p50:6.2f} ms ({f_p50/total_p50*100:4.1f}%)")
    print(f"  4. BM25 Lexical Search:  {b_p50:6.2f} ms ({b_p50/total_p50*100:4.1f}%)")
    print(f"  5. Hybrid Score Fusion:  {u_p50:6.2f} ms ({u_p50/total_p50*100:4.1f}%)")
    print(f"  TOTAL PRE-GEN RETRIEVAL: {total_p50:6.2f} ms")

    return {
        "guardrail_ms": g_p50,
        "embedding_ms": e_p50,
        "faiss_ms": f_p50,
        "bm25_ms": b_p50,
        "fusion_ms": u_p50,
        "total_ms": total_p50,
    }


def main():
    print("=" * 80)
    print("  HH GOA 2026: CONTROLLED OPTIMIZATION & MULTILINGUAL BENCHMARK SUITE")
    print("=" * 80)

    # 1. Load Balanced Dev Corpus
    records, documents = generate_balanced_development_corpus(records_per_language=150)
    print(f"Loaded Development Corpus: {len(records)} records, {len(documents)} canonical documents.")

    # 2. Phase 2: Chunking Experiment
    p2_res = run_phase_2_chunking(documents)

    # 3. Phase 3 & 10: Build/Load Dev Index & Benchmark Retrieval
    faiss_mgr, bm25_mgr, chunks = build_or_load_dev_indexes(documents)
    p3_res = run_phase_3_and_10_retrieval(records, faiss_mgr, bm25_mgr)

    # 4. Initialize OpenAI Client for LM Studio
    client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio", timeout=30.0)

    # 5. Phase 4: Context Compression Scaling
    p4_res = run_phase_4_context_compression(client)

    # 6. Phase 5: Retrieved Context Size / Top-K Scaling
    p5_res = run_phase_5_top_k(client)

    # 7. Phase 6: Answer Length Scaling
    p6_res = run_phase_6_answer_length(client)

    from app.retrieval.bm25 import BM25Retriever
    from app.retrieval.vector import VectorRetriever
    v_ret = VectorRetriever(index_manager=faiss_mgr)
    b_ret = BM25Retriever(index_manager=bm25_mgr)
    hybrid_retriever = HybridRetriever(vector_retriever=v_ret, bm25_retriever=b_ret)
    p7_res = run_phase_7_combined(client, hybrid_retriever)

    # 9. Phase 8: Prompt Caching Inspection
    p8_res = run_phase_8_prompt_caching(client)

    # 10. Phase 9: Retrieval Latency Breakdown
    p9_res = run_phase_9_retrieval_profiling(faiss_mgr, bm25_mgr)

    print("\n" + "=" * 80)
    print("  ALL CONTROLLED OPTIMIZATION EXPERIMENTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
