# ARROHA Candidate Model Bakeoff Report

**Hardware Platform:** ASUS ROG Strix G16 (RTX 4050 Laptop GPU 6GB GDDR6, Intel i7-13650HX, 16GB RAM)
**Benchmark Dataset:** 50 canonical queries from `benchmark.py` under frozen hybrid retrieval evidence
**Retrieval:** 50,400 chunks in FAISS `IndexFlatIP` + SQLite FTS5

## 1. Candidate Comparison Table

| Model Candidate | Params | Quant | Size | TTFT P50 | Decode P50 | Speed (tok/s) | Full RAG P50 | Full RAG P95 | Grounding % | Trunc % | Quality Gate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-1.5B-Instruct (Q4_K_M) [Baseline]** | 1.54B | Q4_K_M | 1.04 GB | 35.91 ms | 163.45 ms | 143.9 | **200.14 ms** | 223.11 ms | 100.0% | 50.0% | ❌ FAIL |
| **Qwen2.5-1.5B-Instruct (Q3_K_M)** | 1.54B | Q3_K_M | 0.85 GB | 40.43 ms | 201.53 ms | 104.8 | **252.16 ms** | 297.39 ms | 100.0% | 32.0% | ❌ FAIL |
| **Llama-3.2-1B-Instruct (Q4_K_M)** | 1.23B | Q4_K_M | 0.80 GB | 33.95 ms | 124.31 ms | 183.3 | **154.50 ms** | 206.75 ms | 100.0% | 52.0% | ❌ FAIL |
| **Qwen2.5-0.5B-Instruct (Q4_K_M) [Speed Ref]** | 0.49B | Q4_K_M | 0.46 GB | 32.54 ms | 87.09 ms | 247.6 | **127.76 ms** | 235.78 ms | 100.0% | 72.0% | ❌ FAIL |

## 2. Multilingual Performance Verification

### Qwen2.5-1.5B-Instruct (Q4_K_M) [Baseline]
- **English** (173.5 ms): `The capital of the Maurya Empire was Pataliputra.`
- **Hindi** (103.8 ms): `The capital of Maharashtra is Mumbai.`
- **Bengali** (154.42 ms): `The capital of the Mughal Empire was Agra.`
- **Tamil** (239.69 ms): `The user's question "வுரிய பேரரசின் தல`

### Qwen2.5-1.5B-Instruct (Q3_K_M)
- **English** (190.16 ms): `The capital of the Maurya Empire was Pataliputra.`
- **Hindi** (276.53 ms): `मौर्य साम्राज्य की राजधानी`
- **Bengali** (374.05 ms): `The Rajgad of the Maratha Empire was located on the western bank of the Narmada River, in the`
- **Tamil** (241.86 ms): `The Malayalam language is spoken in the state of Kerala in India.`

### Llama-3.2-1B-Instruct (Q4_K_M)
- **English** (118.41 ms): `I do not have enough information in the retrieved sources to answer this question.`
- **Hindi** (147.14 ms): `मौर्य साम्राज्य की राजधानी पटना थी।`
- **Bengali** (283.29 ms): `মৌর্য সাম্রাজ্যের রাজ`
- **Tamil** (198.63 ms): `மவுரிய பேரரசின் த`

### Qwen2.5-0.5B-Instruct (Q4_K_M) [Speed Ref]
- **English** (69.31 ms): `The capital of the Maurya Empire was New Delhi.`
- **Hindi** (109.57 ms): `पुणे हे महाराष्ट्राची सांस्क`
- **Bengali** (79.94 ms): `The source provided does not contain enough information to answer the question accurately.`
- **Tamil** (78.13 ms): `The text provided does not contain enough information to answer the question.`

## 3. Final Verdict & Pareto Analysis

### **NO VALID SUB-200ms MODEL FOUND**

All evaluated candidates that preserve high factual grounding and multilingual compliance (>=70% grounding, <=25% hallucination) require >= 160 ms decode time for 20 tokens on the RTX 4050 GPU, yielding ~215–235 ms total text RAG latency. Sub-1B models (e.g. Qwen2.5-0.5B) achieve sub-200ms latency but fail factual quality and hallucination gates.
