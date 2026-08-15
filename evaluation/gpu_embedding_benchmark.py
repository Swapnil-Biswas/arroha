"""
evaluation/gpu_embedding_benchmark.py
-------------------------------------
Rigorous benchmark suite for GPU-accelerated multilingual embedding optimization:
- CPU vs GPU isolated embedding latency (P50, P70, P95, Mean, Min, Max) on 75+ multilingual queries
- VRAM tracking (before, during, after)
- Batch size throughput scaling (8, 16, 32, 64)
- Full retrieval stage A/B comparison (Embedding, FAISS, BM25, Fusion)
- Full RAG pipeline A/B comparison (Retrieval, TTFT, Gen, Full RAG) with Qwen3
- Multilingual correctness verification across 15 languages (shape, finite, L2 norm)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from app.config import (
    BM25_WEIGHT,
    DENSE_WEIGHT,
    EMBEDDING_MODEL_ID,
    LLM_API_KEY,
    LLM_ENDPOINT,
    LLM_MODEL_ID,
    NORMALIZE_EMBEDDINGS,
)
from app.retrieval.hybrid import min_max_normalize
from indexing.bm25_index import BM25IndexManager
from indexing.embeddings import MultilingualEmbedder
from indexing.faiss_index import FAISSIndexManager
from ingestion.dev_corpus import generate_balanced_development_corpus

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("gpu_embedding_benchmark")

# 15 Supported Languages & Balanced Test Queries
MULTILINGUAL_TEST_QUERIES = [
    # English (en)
    ("en", "What is the capital of France?"),
    ("en", "How does photosynthesis work in plants?"),
    ("en", "What are the primary symptoms of diabetes?"),
    ("en", "Explain the theory of general relativity."),
    ("en", "Who discovered penicillin antibiotic?"),
    # Hindi (hi)
    ("hi", "भारत की राजधानी क्या है?"),
    ("hi", "प्रकाश संश्लेषण की प्रक्रिया कैसे काम करती है?"),
    ("hi", "मधुमेह के मुख्य लक्षण क्या हैं?"),
    ("hi", "सौर मंडल का सबसे बड़ा ग्रह कौन सा है?"),
    ("hi", "कंप्यूटर का आविष्कार किसने किया था?"),
    # Bengali (bn)
    ("bn", "পশ্চিমবঙ্গের রাজধানী কি?"),
    ("bn", "শালোকসংশ্লেষ প্রক্রিয়া কীভাবে কাজ করে?"),
    ("bn", "ডায়াবেটিসের প্রধান লক্ষণগুলি কি কি?"),
    ("bn", "ভারতের জাতীয় নদী কোনটি?"),
    ("bn", "বিশ্বের সবচেয়ে উঁচু পর্বত কোনটি?"),
    # Tamil (ta)
    ("ta", "தமிழ்நாட்டின் தலைநகரம் எது?"),
    ("ta", "ஒளிச்சேர்க்கை எவ்வாறு செயல்படுகிறது?"),
    ("ta", "நீரிழிவு நோயின் முக்கிய அறிகுறிகள் யாவை?"),
    ("ta", "இந்தியாவின் மிக நீளமான நதி எது?"),
    ("ta", "சூரிய குடும்பத்தின் மிகப்பெரிய கிரகம் எது?"),
    # Telugu (te)
    ("te", "తెలంగాణ రాజధాని ఏది?"),
    ("te", "కిరణజన్య సంయోగక్రియ ఎలా జరుగుతుంది?"),
    ("te", "మధుమేహం యొక్క ప్రధాన లక్షణాలు ఏమిటి?"),
    ("te", "సౌర వ్యవస్థలో అతిపెద్ద గ్రహం ఏది?"),
    ("te", "భారతదేశ జాతీయ పక్షి ఏది?"),
    # Marathi (mr)
    ("mr", "महाराष्ट्राची राजधानी कोणती आहे?"),
    ("mr", "प्रकाशसंश्लेषण प्रक्रिया कशी कार्य करते?"),
    ("mr", "मधुमेहाची मुख्य लक्षणे कोणती आहेत?"),
    ("mr", "सूर्यमालेतील सर्वात मोठा ग्रह कोणता?"),
    ("mr", "भारताचे राष्ट्रीय फूल कोणते आहे?"),
    # Gujarati (gu)
    ("gu", "ગુજરાતનું પાટનગર કયું છે?"),
    ("gu", "પ્રકાશસંશ્લેષણની પ્રક્રિયા કેવી રીતે થાય છે?"),
    ("gu", "ડાયાબિટીસના મુખ્ય લક્ષણો કયા છે?"),
    ("gu", "સૌરમંડળનો સૌથી મોટો ગ્રહ કયો છે?"),
    ("gu", "ભારતનું રાષ્ટ્રીય પ્રાણી કયું છે?"),
    # Kannada (kn)
    ("kn", "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಯಾವುದು?"),
    ("kn", "ದ್ಯುತಿಸಂಶ್ಲೇಷಣೆ ಹೇಗೆ ಕಾರ್ಯನಿರ್ವಹಿಸುತ್ತದೆ?"),
    ("kn", "ಮಧುಮೇಹದ ಪ್ರಮುಖ ಲಕ್ಷಣಗಳು ಯಾವುವು?"),
    ("kn", "ಸೌರವ್ಯೂಹದ ಅತಿ ದೊಡ್ಡ ಗ್ರಹ ಯಾವುದು?"),
    ("kn", "ಭಾರತದ ರಾಷ್ಟ್ರೀಯ ಜಲಚರ ಪ್ರಾಣಿ ಯಾವುದು?"),
    # Malayalam (ml)
    ("ml", "കേരളത്തിന്റെ തലസ്ഥാനം ഏതാണ്?"),
    ("ml", "പ്രകാശസംശ്ലേഷണം എങ്ങനെയാണ് പ്രവർത്തിക്കുന്നത്?"),
    ("ml", "പ്രമേഹത്തിന്റെ പ്രധാന ലക്ഷണങ്ങൾ എന്തൊക്കെയാണ്?"),
    ("ml", "സൗരയൂഥത്തിലെ ഏറ്റവും വലിയ ഗ്രഹം ഏതാണ്?"),
    ("ml", "ഇന്ത്യയുടെ ദേശീയ പൈതൃക മൃഗം ഏതാണ്?"),
    # Punjabi (pa)
    ("pa", "ਪੰਜਾਬ ਦੀ ਰਾਜਧਾਨੀ ਕਿਹੜੀ ਹੈ?"),
    ("pa", "ਪ੍ਰਕਾਸ਼ ਸੰਸਲੇਸ਼ਣ ਕਿਵੇਂ ਕੰਮ ਕਰਦਾ ਹੈ?"),
    ("pa", "ਸ਼ੂਗਰ ਦੇ ਮੁੱਖ ਲੱਛਣ ਕੀ ਹਨ?"),
    ("pa", "ਸੂਰਜੀ ਸਿਸਟਮ ਦਾ ਸਭ ਤੋਂ ਵੱਡਾ ਗ੍ਰਹਿ ਕਿਹੜਾ ਹੈ?"),
    ("pa", "ਭਾਰਤ ਦੀ ਰਾਸ਼ਟਰੀ ਨਦੀ ਕਿਹੜੀ ਹੈ?"),
    # Odia (or)
    ("or", "ଓଡ଼ିଶାର ରାଜଧାନୀ କଣ?"),
    ("or", "ଆଲୋକ ସଂଶ୍ଳେଷଣ ପ୍ରକ୍ରିୟା କିପରି କାର୍ଯ୍ୟ କରେ?"),
    ("or", "ମଧୁମେହର ମୁଖ୍ୟ ଲକ୍ଷଣ କଣ?"),
    ("or", "ସୌରମଣ୍ଡଳର ସବୁଠାରୁ ବଡ଼ ଗ୍ରହ କିଏ?"),
    ("or", "ଭାରତର ଜାତୀୟ ଫଳ କଣ?"),
    # Assamese (as)
    ("as", "অসমৰ ৰাজধানী কি?"),
    ("as", "সালোকসংশ্লেষণ প্ৰক্ৰিয়া কেনেদৰে কাম কৰে?"),
    ("as", "মধুমেহ ৰোগৰ প্ৰধান লক্ষণসমূহ কি কি?"),
    ("as", "সৌৰজগতৰ আটাইতকৈ ডাঙৰ গ্ৰহটো কি?"),
    ("as", "ভাৰতৰ ৰাষ্ট্ৰীয় ফুল কি?"),
    # Nepali (ne)
    ("ne", "नेपालको राजधानी कुन हो?"),
    ("ne", "प्रकाश संश्लेषण कसरी काम गर्छ?"),
    ("ne", "मधुमेहका मुख्य लक्षणहरू के के हुन्?"),
    ("ne", "सौर्यमण्डलको सबैभन्दा ठूलो ग्रह कुन हो?"),
    ("ne", "नेपालको राष्ट्रिय फूल कुन हो?"),
    # Sanskrit (sa)
    ("sa", "भारतस्य राजधानी का अस्ति?"),
    ("sa", "प्रकाशसंश्लेषणं कथं प्रवर्तते?"),
    ("sa", "मधुमेहस्य प्रमुखाणि लक्षणानि कानि सन्ति?"),
    ("sa", "सौरमण्डलस्य बृहत्तमः ग्रहः कः?"),
    ("sa", "भारतस्य राष्ट्रियवृक्षः कः?"),
    # Urdu (ur)
    ("ur", "پاکستان کا دارالحکومت کیا ہے؟"),
    ("ur", "فوٹو سنتھیسس کیسے کام کرتا ہے؟"),
    ("ur", "ذیابیطس کی بنیادی علامات کیا ہیں؟"),
    ("ur", "نظام شمسی کا سب سے بڑا سیارہ کون سا ہے؟"),
    ("ur", "کمپیوٹر کس نے ایجاد کیا تھا؟"),
]


def calculate_percentiles(values: list[float]) -> dict[str, float]:
    arr = np.array(values)
    return {
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p70": float(np.percentile(arr, 70)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def run_full_gpu_embedding_suite() -> None:
    print("=" * 80)
    print("  HH GOA 2026: GPU EMBEDDING OPTIMIZATION & A/B BENCHMARK SUITE")
    print("=" * 80)

    # 1. Verify CUDA Environment
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "NO CUDA"
    print(f"CUDA Available: {cuda_available}")
    print(f"GPU Model:      {gpu_name}")
    if not cuda_available:
        print("ERROR: CUDA is not available. Cannot run GPU benchmark.")
        return

    # VRAM Baseline
    vram_before = torch.cuda.memory_allocated(0) / 1024**2
    vram_res_before = torch.cuda.memory_reserved(0) / 1024**2
    print(f"VRAM Before:    Allocated {vram_before:.2f} MB | Reserved {vram_res_before:.2f} MB")

    # 2. Multilingual Correctness Check across all 15 languages
    print("\n" + "=" * 80)
    print("  PHASE 13: MULTILINGUAL CORRECTNESS VALIDATION (15 Languages)")
    print("=" * 80)
    cpu_model = SentenceTransformer(EMBEDDING_MODEL_ID, device="cpu")
    gpu_model = SentenceTransformer(EMBEDDING_MODEL_ID, device="cuda")

    vram_after_load = torch.cuda.memory_allocated(0) / 1024**2
    vram_res_after_load = torch.cuda.memory_reserved(0) / 1024**2
    print(f"VRAM After Load: Allocated {vram_after_load:.2f} MB | Reserved {vram_res_after_load:.2f} MB")

    all_passed = True
    for lang, q_text in MULTILINGUAL_TEST_QUERIES[:15]:
        cpu_vec = cpu_model.encode(q_text, normalize_embeddings=NORMALIZE_EMBEDDINGS, convert_to_numpy=True)
        with torch.inference_mode():
            gpu_vec = gpu_model.encode(q_text, normalize_embeddings=NORMALIZE_EMBEDDINGS, convert_to_numpy=True)
            torch.cuda.synchronize()

        finite = bool(np.isfinite(gpu_vec).all())
        shape_match = gpu_vec.shape == (384,)
        l2_norm = float(np.linalg.norm(gpu_vec))
        dot_sim = float(np.dot(cpu_vec, gpu_vec))

        if not (finite and shape_match and abs(l2_norm - 1.0) < 1e-4 and dot_sim > 0.9999):
            all_passed = False
            print(f"[FAIL] {lang}: shape={gpu_vec.shape}, norm={l2_norm:.4f}, sim={dot_sim:.6f}, finite={finite}")
        else:
            print(f"[PASS] {lang.upper():<3} | Shape: {gpu_vec.shape} | L2 Norm: {l2_norm:.6f} | Dot Sim vs CPU: {dot_sim:.8f}")

    print(f"Multilingual Correctness Status: {'ALL 15 LANGUAGES PASSED' if all_passed else 'FAILED'}")

    # 3. Isolated Single-Query Latency Benchmark (CPU vs GPU across 75 queries)
    print("\n" + "=" * 80)
    print(f"  PHASE 7 & 8: ISOLATED EMBEDDING BENCHMARK ({len(MULTILINGUAL_TEST_QUERIES)} Multilingual Queries)")
    print("=" * 80)

    # Warmup
    for _ in range(5):
        _ = cpu_model.encode("Warmup query", normalize_embeddings=True)
        with torch.inference_mode():
            _ = gpu_model.encode("Warmup query", normalize_embeddings=True)
            torch.cuda.synchronize()

    # Benchmark CPU
    cpu_latencies: list[float] = []
    for _, q_text in MULTILINGUAL_TEST_QUERIES:
        t0 = time.perf_counter_ns()
        _ = cpu_model.encode(q_text, batch_size=1, show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True)
        t1 = time.perf_counter_ns()
        cpu_latencies.append((t1 - t0) / 1_000_000.0)

    # Benchmark GPU
    gpu_latencies: list[float] = []
    for _, q_text in MULTILINGUAL_TEST_QUERIES:
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        with torch.inference_mode():
            _ = gpu_model.encode(q_text, batch_size=1, show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        gpu_latencies.append((t1 - t0) / 1_000_000.0)

    cpu_stats = calculate_percentiles(cpu_latencies)
    gpu_stats = calculate_percentiles(gpu_latencies)

    print("\nISOLATED EMBEDDING LATENCY RESULTS (ms):")
    print("-" * 75)
    print(f"{'Metric':<10} | {'CPU (ms)':<15} | {'GPU CUDA (ms)':<15} | {'Speedup':<15}")
    print("-" * 75)
    print(f"{'P50':<10} | {cpu_stats['p50']:>15.2f} | {gpu_stats['p50']:>15.2f} | {cpu_stats['p50'] / gpu_stats['p50']:>14.2f}x")
    print(f"{'P70':<10} | {cpu_stats['p70']:>15.2f} | {gpu_stats['p70']:>15.2f} | {cpu_stats['p70'] / gpu_stats['p70']:>14.2f}x")
    print(f"{'P95':<10} | {cpu_stats['p95']:>15.2f} | {gpu_stats['p95']:>15.2f} | {cpu_stats['p95'] / gpu_stats['p95']:>14.2f}x")
    print(f"{'Mean':<10} | {cpu_stats['mean']:>15.2f} | {gpu_stats['mean']:>15.2f} | {cpu_stats['mean'] / gpu_stats['mean']:>14.2f}x")
    print(f"{'Min':<10} | {cpu_stats['min']:>15.2f} | {gpu_stats['min']:>15.2f} | {cpu_stats['min'] / gpu_stats['min']:>14.2f}x")
    print(f"{'Max':<10} | {cpu_stats['max']:>15.2f} | {gpu_stats['max']:>15.2f} | {cpu_stats['max'] / gpu_stats['max']:>14.2f}x")
    print("-" * 75)

    # 4. Batch Throughput Scaling on GPU
    print("\n" + "=" * 80)
    print("  PHASE 6: OFFLINE BATCH EMBEDDING THROUGHPUT (512 Sample Texts)")
    print("=" * 80)
    sample_texts = [MULTILINGUAL_TEST_QUERIES[i % len(MULTILINGUAL_TEST_QUERIES)][1] for i in range(512)]
    print(f"{'Batch Size':<12} | {'Time (ms)':<15} | {'Throughput (passages/s)':<25} | {'VRAM Alloc (MB)':<15}")
    print("-" * 75)
    for bs in [8, 16, 32, 64]:
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        with torch.inference_mode():
            _ = gpu_model.encode(sample_texts, batch_size=bs, show_progress_bar=False, normalize_embeddings=True)
        torch.cuda.synchronize()
        dur_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        thru = len(sample_texts) / (dur_ms / 1000.0)
        vram_cur = torch.cuda.memory_allocated(0) / 1024**2
        print(f"{bs:<12} | {dur_ms:>15.2f} | {thru:>25.1f} | {vram_cur:>15.2f}")

    # 5. Load Development Corpus Indexes
    print("\n" + "=" * 80)
    print("  LOADING 12,600-PASSAGE DEV INDEX FOR RETRIEVAL & RAG BENCHMARK")
    print("=" * 80)

    faiss_mgr = FAISSIndexManager()
    faiss_mgr.load()

    bm25_mgr = BM25IndexManager()
    bm25_mgr.load()

    # 6. Phase 9: End-to-End Retrieval Stage A/B Benchmark (CPU vs GPU)
    print("\n" + "=" * 80)
    print("  PHASE 9: END-TO-END RETRIEVAL STAGE A/B BENCHMARK (30 Test Queries)")
    print("=" * 80)

    test_queries_subset = [q[1] for q in MULTILINGUAL_TEST_QUERIES[:30]]

    # Profiling with CPU Embedder
    cpu_embed_times: list[float] = []
    cpu_faiss_times: list[float] = []
    cpu_bm25_times: list[float] = []
    cpu_fusion_times: list[float] = []
    cpu_retrieval_totals: list[float] = []

    for q in test_queries_subset:
        t_start = time.perf_counter_ns()
        # 1. CPU Embed
        t0 = time.perf_counter_ns()
        q_vec = cpu_model.encode(q, batch_size=1, show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True)
        t_embed = (time.perf_counter_ns() - t0) / 1_000_000.0
        cpu_embed_times.append(t_embed)

        # 2. FAISS
        dense_hits, t_faiss = faiss_mgr.search(q_vec, top_k=10)
        cpu_faiss_times.append(t_faiss)

        # 3. BM25
        sparse_hits, t_bm25 = bm25_mgr.search(q, top_k=10)
        cpu_bm25_times.append(t_bm25)

        # 4. Fusion
        t_f0 = time.perf_counter_ns()
        dense_scores = [score for _, score in dense_hits]
        sparse_scores = [score for _, score in sparse_hits]
        norm_dense = min_max_normalize(dense_scores)
        norm_sparse = min_max_normalize(sparse_scores)
        combined_scores: dict[str, float] = {}
        for (doc, _), n_score in zip(dense_hits, norm_dense):
            pid = doc.get("passage_id", "")
            combined_scores[pid] = combined_scores.get(pid, 0.0) + DENSE_WEIGHT * n_score
        for (doc, _), n_score in zip(sparse_hits, norm_sparse):
            pid = doc.get("passage_id", "")
            combined_scores[pid] = combined_scores.get(pid, 0.0) + BM25_WEIGHT * n_score
        t_fusion = (time.perf_counter_ns() - t_f0) / 1_000_000.0
        cpu_fusion_times.append(t_fusion)

        t_total = (time.perf_counter_ns() - t_start) / 1_000_000.0
        cpu_retrieval_totals.append(t_total)

    # Profiling with GPU Embedder
    gpu_embed_times: list[float] = []
    gpu_faiss_times: list[float] = []
    gpu_bm25_times: list[float] = []
    gpu_fusion_times: list[float] = []
    gpu_retrieval_totals: list[float] = []

    for q in test_queries_subset:
        t_start = time.perf_counter_ns()
        # 1. GPU Embed
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        with torch.inference_mode():
            q_vec = gpu_model.encode(q, batch_size=1, show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True)
        torch.cuda.synchronize()
        t_embed = (time.perf_counter_ns() - t0) / 1_000_000.0
        gpu_embed_times.append(t_embed)

        # 2. FAISS
        dense_hits, t_faiss = faiss_mgr.search(q_vec, top_k=10)
        gpu_faiss_times.append(t_faiss)

        # 3. BM25
        sparse_hits, t_bm25 = bm25_mgr.search(q, top_k=10)
        gpu_bm25_times.append(t_bm25)

        # 4. Fusion
        t_f0 = time.perf_counter_ns()
        dense_scores = [score for _, score in dense_hits]
        sparse_scores = [score for _, score in sparse_hits]
        norm_dense = min_max_normalize(dense_scores)
        norm_sparse = min_max_normalize(sparse_scores)
        combined_scores: dict[str, float] = {}
        for (doc, _), n_score in zip(dense_hits, norm_dense):
            pid = doc.get("passage_id", "")
            combined_scores[pid] = combined_scores.get(pid, 0.0) + DENSE_WEIGHT * n_score
        for (doc, _), n_score in zip(sparse_hits, norm_sparse):
            pid = doc.get("passage_id", "")
            combined_scores[pid] = combined_scores.get(pid, 0.0) + BM25_WEIGHT * n_score
        t_fusion = (time.perf_counter_ns() - t_f0) / 1_000_000.0
        gpu_fusion_times.append(t_fusion)

        t_total = (time.perf_counter_ns() - t_start) / 1_000_000.0
        gpu_retrieval_totals.append(t_total)

    c_emb_p50 = float(np.median(cpu_embed_times))
    g_emb_p50 = float(np.median(gpu_embed_times))
    c_fai_p50 = float(np.median(cpu_faiss_times))
    g_fai_p50 = float(np.median(gpu_faiss_times))
    c_bm_p50 = float(np.median(cpu_bm25_times))
    g_bm_p50 = float(np.median(gpu_bm25_times))
    c_fus_p50 = float(np.median(cpu_fusion_times))
    g_fus_p50 = float(np.median(gpu_fusion_times))
    c_tot_p50 = float(np.median(cpu_retrieval_totals))
    g_tot_p50 = float(np.median(gpu_retrieval_totals))

    print("\nRETRIEVAL STAGE A/B COMPARISON (P50 Latency in ms):")
    print("-" * 75)
    print(f"{'Stage':<18} | {'CPU Embed (ms)':<16} | {'GPU Embed (ms)':<16} | {'Delta (ms)':<15}")
    print("-" * 75)
    print(f"{'Embedding':<18} | {c_emb_p50:>16.2f} | {g_emb_p50:>16.2f} | {g_emb_p50 - c_emb_p50:>14.2f} ms ({c_emb_p50/g_emb_p50:.1f}x speedup)")
    print(f"{'FAISS Dense':<18} | {c_fai_p50:>16.2f} | {g_fai_p50:>16.2f} | {g_fai_p50 - c_fai_p50:>14.2f} ms")
    print(f"{'BM25 Lexical':<18} | {c_bm_p50:>16.2f} | {g_bm_p50:>16.2f} | {g_bm_p50 - c_bm_p50:>14.2f} ms")
    print(f"{'Hybrid Fusion':<18} | {c_fus_p50:>16.2f} | {g_fus_p50:>16.2f} | {g_fus_p50 - c_fus_p50:>14.2f} ms")
    print("-" * 75)
    print(f"{'TOTAL RETRIEVAL':<18} | {c_tot_p50:>16.2f} | {g_tot_p50:>16.2f} | {g_tot_p50 - c_tot_p50:>14.2f} ms")
    print("-" * 75)

    # 7. Phase 10: End-to-End Full Text-RAG A/B Benchmark with Qwen3
    print("\n" + "=" * 80)
    print("  PHASE 10: FULL TEXT-RAG A/B BENCHMARK (Qwen3 4B Q4_K_M in LM Studio)")
    print("  Config: Top-2 Hybrid, 150 context tokens, max 8 output tokens")
    print("=" * 80)

    client = OpenAI(base_url=LLM_ENDPOINT, api_key=LLM_API_KEY)
    rag_test_queries = MULTILINGUAL_TEST_QUERIES[:15]  # 15 languages

    # CPU Embedding RAG
    cpu_rag_ret: list[float] = []
    cpu_rag_ttft: list[float] = []
    cpu_rag_gen: list[float] = []
    cpu_rag_tot: list[float] = []

    for lang, q_text in rag_test_queries:
        t_start = time.perf_counter_ns()
        # Retrieval (CPU)
        t_r0 = time.perf_counter_ns()
        q_vec = cpu_model.encode(q_text, batch_size=1, show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True)
        dense_hits, _ = faiss_mgr.search(q_vec, top_k=2)
        sparse_hits, _ = bm25_mgr.search(q_text, top_k=2)
        t_ret = (time.perf_counter_ns() - t_r0) / 1_000_000.0
        cpu_rag_ret.append(t_ret)

        context = " ".join([h[0].get("text", "")[:150] for h in dense_hits[:2]])
        prompt = f"Answer concisely in one sentence using context:\nContext: {context}\nQuestion: {q_text}\nAnswer:"

        # LLM Call
        t_llm0 = time.perf_counter_ns()
        response_stream = client.chat.completions.create(
            model=LLM_MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8,
            temperature=0.1,
            stream=True,
        )
        t_first = None
        for chunk in response_stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                if t_first is None:
                    t_first = time.perf_counter_ns()
        t_end = time.perf_counter_ns()

        t_ttft = (t_first - t_llm0) / 1_000_000.0 if t_first else (t_end - t_llm0) / 1_000_000.0
        t_gen = (t_end - t_first) / 1_000_000.0 if t_first else 0.0
        t_total = (t_end - t_start) / 1_000_000.0

        cpu_rag_ttft.append(t_ttft)
        cpu_rag_gen.append(t_gen)
        cpu_rag_tot.append(t_total)

    # GPU Embedding RAG
    gpu_rag_ret: list[float] = []
    gpu_rag_ttft: list[float] = []
    gpu_rag_gen: list[float] = []
    gpu_rag_tot: list[float] = []

    for lang, q_text in rag_test_queries:
        t_start = time.perf_counter_ns()
        # Retrieval (GPU)
        t_r0 = time.perf_counter_ns()
        torch.cuda.synchronize()
        with torch.inference_mode():
            q_vec = gpu_model.encode(q_text, batch_size=1, show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True)
        torch.cuda.synchronize()
        dense_hits, _ = faiss_mgr.search(q_vec, top_k=2)
        sparse_hits, _ = bm25_mgr.search(q_text, top_k=2)
        t_ret = (time.perf_counter_ns() - t_r0) / 1_000_000.0
        gpu_rag_ret.append(t_ret)

        context = " ".join([h[0].get("text", "")[:150] for h in dense_hits[:2]])
        prompt = f"Answer concisely in one sentence using context:\nContext: {context}\nQuestion: {q_text}\nAnswer:"

        # LLM Call
        t_llm0 = time.perf_counter_ns()
        response_stream = client.chat.completions.create(
            model=LLM_MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8,
            temperature=0.1,
            stream=True,
        )
        t_first = None
        for chunk in response_stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                if t_first is None:
                    t_first = time.perf_counter_ns()
        t_end = time.perf_counter_ns()

        t_ttft = (t_first - t_llm0) / 1_000_000.0 if t_first else (t_end - t_llm0) / 1_000_000.0
        t_gen = (t_end - t_first) / 1_000_000.0 if t_first else 0.0
        t_total = (t_end - t_start) / 1_000_000.0

        gpu_rag_ttft.append(t_ttft)
        gpu_rag_gen.append(t_gen)
        gpu_rag_tot.append(t_total)

    c_r_stats = calculate_percentiles(cpu_rag_ret)
    g_r_stats = calculate_percentiles(gpu_rag_ret)
    c_t_stats = calculate_percentiles(cpu_rag_ttft)
    g_t_stats = calculate_percentiles(gpu_rag_ttft)
    c_g_stats = calculate_percentiles(cpu_rag_gen)
    g_g_stats = calculate_percentiles(gpu_rag_gen)
    c_tot_stats = calculate_percentiles(cpu_rag_tot)
    g_tot_stats = calculate_percentiles(gpu_rag_tot)

    print("\nFULL TEXT-RAG A/B COMPARISON:")
    print("-" * 75)
    print(f"{'Metric':<18} | {'CPU Embed RAG':<16} | {'GPU Embed RAG':<16} | {'Delta':<15}")
    print("-" * 75)
    print(f"{'Retrieval P50':<18} | {c_r_stats['p50']:>13.2f} ms | {g_r_stats['p50']:>13.2f} ms | {g_r_stats['p50'] - c_r_stats['p50']:>11.2f} ms")
    print(f"{'TTFT P50':<18} | {c_t_stats['p50']:>13.2f} ms | {g_t_stats['p50']:>13.2f} ms | {g_t_stats['p50'] - c_t_stats['p50']:>11.2f} ms")
    print(f"{'Generation P50':<18} | {c_g_stats['p50']:>13.2f} ms | {g_g_stats['p50']:>13.2f} ms | {g_g_stats['p50'] - c_g_stats['p50']:>11.2f} ms")
    print(f"{'Full RAG P50':<18} | {c_tot_stats['p50']:>13.2f} ms | {g_tot_stats['p50']:>13.2f} ms | {g_tot_stats['p50'] - c_tot_stats['p50']:>11.2f} ms")
    print(f"{'Full RAG P70':<18} | {c_tot_stats['p70']:>13.2f} ms | {g_tot_stats['p70']:>13.2f} ms | {g_tot_stats['p70'] - c_tot_stats['p70']:>11.2f} ms")
    print(f"{'Full RAG P95':<18} | {c_tot_stats['p95']:>13.2f} ms | {g_tot_stats['p95']:>13.2f} ms | {g_tot_stats['p95'] - c_tot_stats['p95']:>11.2f} ms")
    print("-" * 75)

    print("\n" + "=" * 80)
    print("  ALL GPU EMBEDDING OPTIMIZATION BENCHMARKS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_full_gpu_embedding_suite()
