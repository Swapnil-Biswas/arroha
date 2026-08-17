# ARROHA End-to-End Full Text RAG Forensic Latency Benchmark

**LLM Candidate:** `Qwen2.5-1.5B-Instruct Q4_K_M` (`llama-server` CUDA b10451, `-ngl 99`, `--cache-reuse 64`)
**Embedding Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-d, FP16 CUDA hot-path)
**Retrieval Corpus:** 50,400 chunks in FAISS `IndexFlatIP` + SQLite FTS5
**Benchmark Dataset:** 50 canonical queries from `benchmark.py`

## 1. Stage-by-Stage Latency Breakdown (50 Queries)

| Pipeline Stage | Mean | P50 | P95 | P99 | Share of Total |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Query Preprocessing / Input Guardrails** | 0.14 ms | **0.07 ms** | 0.51 ms | 1.19 ms | 0.1% |
| **Query Embedding (FP16 CUDA)** | 43.40 ms | **12.75 ms** | 237.41 ms | 282.72 ms | 15.1% |
| **FAISS Search (IndexFlatIP)** | 0.25 ms | **0.07 ms** | 1.37 ms | 2.61 ms | 0.1% |
| **SQLite BM25 Search** | 0.48 ms | **0.19 ms** | 1.80 ms | 4.12 ms | 0.2% |
| **Hybrid Score Fusion** | 0.12 ms | **0.06 ms** | 0.47 ms | 0.71 ms | 0.0% |
| **Total Retrieval Stage** | 44.62 ms | **13.46 ms** | 240.77 ms | 286.34 ms | 15.5% |
| **Prompt Assembly** | 0.02 ms | **0.01 ms** | 0.06 ms | 0.21 ms | 0.0% |
| **LLM Time-To-First-Token (TTFT)** | 69.98 ms | **52.42 ms** | 205.62 ms | 350.89 ms | 24.4% |
| **LLM T3 (3 Tokens Emitted)** | 91.03 ms | **70.42 ms** | 231.06 ms | 398.77 ms | 31.7% |
| **LLM T5 (5 Tokens Emitted)** | 107.22 ms | **85.74 ms** | 250.60 ms | 419.87 ms | 37.3% |
| **LLM Pure Token Generation / Decode** | 172.27 ms | **169.65 ms** | 322.40 ms | 355.93 ms | 60.0% |
| **Total LLM Stage (TTFT + Decode)** | 242.25 ms | **223.20 ms** | 458.28 ms | 548.95 ms | 84.3% |
| **Output Sanitization & Grounding Check** | 0.18 ms | **0.07 ms** | 0.83 ms | 1.62 ms | 0.1% |
| **TOTAL END-TO-END RAG PIPELINE** | 287.23 ms | **246.23 ms** | 538.20 ms | 591.42 ms | 100.0% |

## 2. Key Forensic Findings & Remaining Bottleneck

- **Total Retrieval Latency:** **13.46 ms P50** (15.5% of total pipeline)
- **Total LLM Generation Latency:** **223.20 ms P50** (84.3% of total pipeline)
  - **TTFT (Prefill / Time to 1st Token):** **52.42 ms P50**
  - **Decode (Token Generation):** **169.65 ms P50**
- **COMPLETE End-to-End Full RAG Latency:** **246.23 ms P50** (P95: **538.20 ms**)
- **Factual Grounding Accuracy:** **68.0%**
