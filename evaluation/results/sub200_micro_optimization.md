# ARROHA Sub-200ms Micro-Optimization Report

**Target LLM:** `Qwen2.5-1.5B-Instruct Q4_K_M`
**Hardware Platform:** ASUS ROG Strix G16 (RTX 4050 6GB GDDR6, Intel i7-13650HX, 16GB RAM)
**Benchmark Scope:** 50 canonical benchmark queries under frozen hybrid retrieval evidence

## 1. Micro-Optimization Summary Table

| Condition | Repetitions | TTFT P50 | Decode P50 | Full RAG P50 | Full RAG P95 | <200ms % | Grounding % | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline (-c 2048)** | 1 | 42.22 ms | 149.62 ms | **190.47 ms** | 352.64 ms | 54.0% | 100.0% | ✅ **PASS (<200ms)** |
| **2. FlashAttention-2 (-fa on, -c 2048)** | 1 | 45.77 ms | 164.32 ms | **217.15 ms** | 364.29 ms | 46.0% | 100.0% | ❌ **FAIL (>200ms)** |
| **3. FlashAttention + Context 1024 + UBatch 512** | 1 | 41.66 ms | 167.63 ms | **209.26 ms** | 281.74 ms | 44.0% | 100.0% | ❌ **FAIL (>200ms)** |
| **4. Low-Latency Polling (poll 100, t=8, fa on)** | 1 | 38.96 ms | 165.91 ms | **207.23 ms** | 419.71 ms | 40.0% | 100.0% | ❌ **FAIL (>200ms)** |
| **5. STABILITY TEST: 1. Baseline (-c 2048) (150 Queries)** | 3 | 43.40 ms | 167.40 ms | **219.23 ms** | 387.70 ms | 33.3% | 100.0% | ❌ **FAIL (>200ms)** |

## 2. Final Verdict

### **SUB-200ms NOT ACHIEVED — CURRENT CONFIGURATION IS OPTIMAL**

- **Steady-State Full RAG P50:** **219.23 ms**
- **P95 Full RAG Latency:** **387.70 ms**
- **Fraction of Requests < 200ms:** **33.3%**
- **Factual Grounding Compliance:** **100.0%**
