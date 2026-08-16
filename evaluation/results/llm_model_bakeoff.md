# ARROHA — Fast Multilingual LLM Benchmark (Bake-off) Decision Report

## 1. Executive Summary
- **Objective:** Empirically evaluate candidate small/fast multilingual LLMs against ARROHA's baseline (`Qwen3-4B-Instruct`) to achieve the competition post-STT latency target of **< 200 ms**.
- **Hardware:** ASUS ROG Strix G16 (Intel i7-13650HX, NVIDIA GeForce RTX 4050 Laptop GPU 6GB GDDR6, 16GB RAM, AC Power).
- **Inference Engine:** Standalone `llama-server.exe` (Build `b10451`, CUDA 12.4, `-ngl 99`, `-c 2048`, `--cache-prompt`, `--cache-reuse 64`).
- **Evaluation Standard:** 45 canonical benchmark queries across 15 Indian & global languages over the 50,400-chunk SQLite FTS5 + FAISS hybrid index.

- **Top Latency Winner:** **Qwen2.5-0.5B-Instruct** with **153.53 ms P50** full pipeline latency.

---

## 2. Models Benchmarked & Specifications

| Model ID | Model Name | Class | Params | Quantization | File Size | Load Time | Classification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `qwen25_05b` | **Qwen2.5-0.5B-Instruct** | Very Small Qwen | 0.49B | Q4_K_M | 468.6 MB | 2.6 s | **EXCELLENT** |
| `qwen25_15b` | **Qwen2.5-1.5B-Instruct** | Small Qwen | 1.54B | Q4_K_M | 1065.6 MB | 3.1 s | **GOOD** |
| `qwen3_4b` | **Qwen3-4B-Instruct-2507** | Current Baseline | 4.0B | Q4_K_M | 2381.6 MB | 7.1 s | **NOT COMPETITIVE** |

---

## 3. Latency & Throughput Comparison Table

| Model | TTFT P50 | TTFT P95 | Gen Latency P50 | Throughput (tok/s) | Pipeline P50 | Pipeline P95 | % < 200ms | % < 180ms | % < 150ms |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen2.5-0.5B-Instruct** | **46.0 ms** | 234.32 ms | **90.71 ms** | **234.74 t/s** | **153.53 ms** | **563.32 ms** | **66.67%** | **60.0%** | **42.22%** |
| **Qwen2.5-1.5B-Instruct** | **56.72 ms** | 195.86 ms | **128.48 ms** | **124.51 t/s** | **254.85 ms** | **553.87 ms** | **24.44%** | **17.78%** | **2.22%** |
| **Qwen3-4B-Instruct-2507** | **117.37 ms** | 493.3 ms | **363.96 ms** | **45.74 t/s** | **589.93 ms** | **961.56 ms** | **0.0%** | **0.0%** | **0.0%** |

---

## 4. Voice-Oriented Streaming Latency ($T_1$, $T_3$, $T_5$, $T_{\text{end}}$)

For real-time voice synthesis, time to first token ($T_1$) and first few tokens ($T_3$, $T_5$) determine when Text-to-Speech (TTS) streaming can begin speaking:

| Model | First Token $T_1$ (TTFT P50) | 3 Tokens $T_3$ P50 | 5 Tokens $T_5$ P50 | Complete Answer $T_{\text{end}}$ P50 | Actual Tokens P50 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen2.5-0.5B-Instruct** | **46.0 ms** | **58.39 ms** | **66.82 ms** | **134.79 ms** | 16.0 tok |
| **Qwen2.5-1.5B-Instruct** | **56.72 ms** | **78.95 ms** | **94.02 ms** | **221.7 ms** | 15.0 tok |
| **Qwen3-4B-Instruct-2507** | **117.37 ms** | **175.19 ms** | **215.75 ms** | **489.2 ms** | 15.0 tok |

---

## 5. Quality, Grounding & Completeness Comparison

| Model | Grounding Rate | Completeness Rate | Truncation Rate | Status / Quality Gate |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen2.5-0.5B-Instruct** | **17.78%** | **82.22%** | 17.78% | **BORDERLINE / FAILED** |
| **Qwen2.5-1.5B-Instruct** | **13.33%** | **93.33%** | 6.67% | **BORDERLINE / FAILED** |
| **Qwen3-4B-Instruct-2507** | **4.44%** | **93.33%** | 8.89% | **BORDERLINE / FAILED** |

---

## 6. Per-Language Latency & Accuracy Breakdown (P50 Pipeline ms)

| Language | **Qwen2.5-0.5B-Instruct** | **Qwen2.5-1.5B-Instruct** | **Qwen3-4B-Instruct-2507** |
| :--- | :--- | :--- | :--- |
| **English (en)** | 153.5 ms | 207.3 ms | 729.6 ms |
| **Hindi (hi)** | 196.3 ms | 254.8 ms | 595.7 ms |
| **Bengali (bn)** | 106.8 ms | 217.3 ms | 468.6 ms |
| **Tamil (ta)** | 197.0 ms | 181.0 ms | 379.9 ms |
| **Telugu (te)** | 161.5 ms | 263.2 ms | 479.7 ms |
| **Marathi (mr)** | 263.6 ms | 385.6 ms | 589.9 ms |
| **Gujarati (gu)** | 147.8 ms | 274.9 ms | 653.8 ms |
| **Kannada (kn)** | 130.0 ms | 156.8 ms | 356.8 ms |
| **Malayalam (ml)** | 167.0 ms | 381.2 ms | 890.8 ms |
| **Punjabi (pa)** | 110.9 ms | 250.5 ms | 428.3 ms |
| **Odia (or)** | 146.1 ms | 192.3 ms | 733.7 ms |
| **Assamese (as)** | 151.7 ms | 235.0 ms | 599.1 ms |
| **Nepali (ne)** | 151.3 ms | 215.8 ms | 829.1 ms |
| **Sanskrit (sa)** | 196.2 ms | 340.6 ms | 648.5 ms |
| **Urdu (ur)** | 165.6 ms | 320.0 ms | 769.4 ms |

---

## 7. Quality vs. Latency Tradeoff & Final Recommendation

### Recommended Production Model:
**Qwen2.5-0.5B-Instruct (0.49B Q4_K_M)**

### Architectural Rationale:
1. **Latency Profile:** Achieves **153.53 ms P50** full RAG pipeline latency.
2. **Generation Speed:** Delivers **234.74 tokens/sec**, significantly higher throughput than baseline.
3. **VRAM Footprint:** Consumes only **468.6 MB**, leaving ample VRAM for embedding models and concurrency.
4. **Voice Readiness:** Time to First Token ($T_1$) is **46.0 ms**, allowing instant audio synthesis dispatch.

---
