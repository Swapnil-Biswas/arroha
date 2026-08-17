# ARROHA LLM Decode Optimization & Quality Forensics

**Target LLM:** `Qwen2.5-1.5B-Instruct Q4_K_M`
**Hardware Platform:** ASUS ROG Strix G16 (RTX 4050 6GB GDDR6, Intel i7-13650HX)
**Benchmark Dataset:** 50 canonical benchmark queries

## 1. Controlled Experiment Summary Table

| Condition | TTFT P50 | Decode P50 | Full RAG P50 | Full RAG P95 | Grounding % | Trunc % | Quality Gate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Condition 1: Baseline (Qwen2.5-1.5B, -c 2048, max_tokens=24)** | 42.43 ms | 171.69 ms | **231.87 ms** | 424.35 ms | 100.0% | 44.0% | ❌ FAIL |
| **Condition 2: Context Optimization (-c 1024, -ub 512)** | 52.51 ms | 172.65 ms | **231.21 ms** | 536.71 ms | 100.0% | 44.0% | ❌ FAIL |
| **Condition 3: Speculative Decoding (1.5B + 0.5B Draft Model)** | 55.68 ms | 195.27 ms | **288.61 ms** | 534.67 ms | 100.0% | 42.0% | ❌ FAIL |
| **Condition 4: Concise Prompt Engineering (-c 1024, max_tokens=24)** | 61.64 ms | 181.36 ms | **273.00 ms** | 546.45 ms | 100.0% | 20.0% | ❌ FAIL |
| **Condition 5: Speculative Decoding + Concise Prompt** | 52.64 ms | 172.78 ms | **244.58 ms** | 433.54 ms | 100.0% | 18.0% | ❌ FAIL |

## 2. Key Findings & Recommended Configuration

- **Best Quality-Preserving Configuration:** **Condition 1: Baseline (Qwen2.5-1.5B, -c 2048, max_tokens=24)**
- **Full RAG P50 Latency:** **231.87 ms** (Sub-200ms Goal Met: `False`)
- **Grounding Accuracy:** **100.0%**
- **Truncation Rate:** **44.0%**
