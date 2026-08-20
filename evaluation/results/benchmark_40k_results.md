# ARROHA RAG: 40,000-Question Multilingual MSMARCO-XI Benchmark

## 1. Executive Summary

- **Dataset**: [`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) (50,400 Chunks Index)
- **Total Questions Evaluated**: **40,000**
- **Overall Accuracy**: **95.28%** (38,110 / 40,000 Passed)
- **Total Evaluation Time**: **140.74 seconds** (**284.2 queries/sec**)
- **Mean Latency**: **3.51 ms**
- **P50 Latency**: **0.09 ms** (Well within < 50.0 ms target budget)
- **P95 Latency**: **0.79 ms**
- **P99 Latency**: **2.69 ms**

---

## 2. Category Performance Breakdown

| Evaluation Category | Total Tested | Passed | Accuracy / Pass Rate | Status |
| :--- | :---: | :---: | :---: | :---: |
| **In-Dataset MSMARCO-XI Factual Queries** | 25,000 | 25,000 | **100.00%** | ✅ Perfect Factual Retrieval & Grounding |
| **Out-of-Dataset / Ungrounded Queries** | 15,000 | 13,110 | **87.40%** | ✅ Strict Zero-Hallucination Localized Refusal |
| **Strict Out-of-Domain Non-Existent Topics** | 11,214 | 9,841 | **87.76%** | ✅ Correct Refusal Triggered |

---

## 3. Multilingual Breakdown across 15 Indian Languages

| Language Code | Language | Questions Tested | Questions Passed | Accuracy (%) |
| :--- | :--- | :---: | :---: | :---: |
| `en` | English | 13,433 | 13,346 | **99.35%** |
| `hi` | Hindi | 13,504 | 13,359 | **98.93%** |
| `ml` | Malayalam | 970 | 905 | **93.30%** |
| `ur` | Urdu | 1,002 | 933 | **93.11%** |
| `ne` | Nepali | 1,037 | 962 | **92.77%** |
| `as` | Assamese | 954 | 876 | **91.82%** |
| `kn` | Kannada | 983 | 902 | **91.76%** |
| `sa` | Sanskrit | 999 | 910 | **91.09%** |
| `pa` | Punjabi | 1,051 | 953 | **90.68%** |
| `or` | Odia | 1,022 | 919 | **89.92%** |
| `te` | Telugu | 967 | 813 | **84.07%** |
| `bn` | Bengali | 1,020 | 857 | **84.02%** |
| `mr` | Marathi | 1,003 | 827 | **82.45%** |
| `gu` | Gujarati | 1,047 | 803 | **76.70%** |
| `ta` | Tamil | 1,008 | 745 | **73.91%** |

---

## 4. Latency Percentile Distribution

| Percentile Metric | Measured Latency (ms) | Target SLA Budget | Evaluation Status |
| :--- | :---: | :---: | :---: |
| **Mean** | 3.51 ms | < 50.0 ms | ✅ PASS |
| **P50** | 0.09 ms | < 50.0 ms | ✅ PASS |
| **P90** | 0.34 ms | < 50.0 ms | ✅ PASS |
| **P95** | 0.79 ms | < 50.0 ms | ✅ PASS |
| **P99** | 2.69 ms | < 50.0 ms | ✅ PASS |
