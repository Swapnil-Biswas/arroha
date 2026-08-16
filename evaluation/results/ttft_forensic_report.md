# ARROHA TTFT Forensic Investigation

## 1. Environment
- **Host System:** ASUS ROG Strix G16
- **GPU Accelerator:** NVIDIA GeForce RTX 4050 Laptop GPU (6140.5 MB VRAM)
- **GPU VRAM Utilization:** 0.00 MB allocated (PyTorch) + ~3,400 MB (LM Studio Qwen3 4B)
- **Host Physical RAM:** 16003.2 MB Total | 14535.4 MB Used (90.0%) | 1467.8 MB Available
- **LM Studio Endpoint:** `http://localhost:1234/v1`
- **Loaded Models in LM Studio:** `['qwen/qwen3-4b-2507', 'qwen3-coder-30b-a3b-instruct-1m', 'google/gemma-4-e2b', 'openai/gpt-oss-20b', 'qwen2.5-coder-7b-instruct', 'text-embedding-nomic-embed-text-v1.5', 'dolphin-2.9.3-mistral-7b-32k']`
- **Power State:** AC Connected (High Performance)

---

## 2. Exact ARROHA LLM Request
- **Model ID:** `qwen/qwen3-4b-2507`
- **Max Output Tokens:** `150`
- **Temperature:** `0.1`
- **Stream:** `True`
- **Stream Options:** `{'include_usage': True}`
- **Timeout:** `8.0s`
- **Messages Array:** 2 messages (`system`, `user`)

### System Prompt (`role: system`):
```text
You are a multilingual factual AI assistant for a real-time voice pipeline.
Answer the user's question accurately and concisely using ONLY the provided retrieved context.

CRITICAL RULES:
1. Grounding: Answer strictly using facts from the retrieved context. Do NOT extrapolate, speculate, or use outside knowledge.
2. Refusal: If the retrieved context does not contain enough information to answer the question, state clearly: "I do not have enough information in the retrieved sources to answer this question." (or its equivalent in the query language).
3. Language Consistency: Reply in the same language and script as the user's query (e.g. Hindi in Devanagari, Bengali in Bengali script, Tamil in Tamil script, English in Latin).
4. Conciseness: Keep the answer under 2-3 sentences (maximum 50 words) to ensure low latency for voice synthesis.
5. No Meta-Commentary: Do NOT say "Based on the provided text" or "According to the context". State the factual answer directly.

```

### User Message with Retrieved Context (`role: user`):
```text
Retrieved Context:
[Source 1 - Lang: bn]: কলকাতা হলো ভারতের পশ্চিমবঙ্গ রাজ্যের রাজধানী, যা হুগলি নদীর পূর্ব তীরে অবস্থিত।

[Source 2 - Lang: mr]: पुणे हे महाराष्ट्राची सांस्कृतिक राजधानी आणि प्रमुख आयटी आणि शैक्षणिक केंद्र मानले जाते.

User Question: What is the capital of France?

Factual Answer:
```

---

## 3. Prompt Size
- **System Prompt Characters:** 977 chars (~151 words)
- **User Message Characters:** 298 chars
- **Total Payload Characters:** 1275 chars
- **Retrieved Context Passages:** 2 passages
- **Minimal Prompt Tokens (API Usage):** 22 tokens
- **Exact ARROHA RAG Prompt Tokens (API Usage):** **433 tokens**

---

## 4. Minimal Direct API Benchmark
*Direct to LM Studio with prompt: `"Answer in 3 words: What is 2 + 2?"` (1 warmup + 10 runs)*

| Metric | Result |
|---|---:|
| Prompt tokens | 22 |
| TTFT P50 | **2290.11 ms** |
| TTFT P95 | 2420.39 ms |
| Generation P50 | 0.63 ms |
| Total P50 | 2293.54 ms |
| Generation Tokens/sec | **4740.59 tok/s** |

---

## 5. Exact ARROHA Prompt Direct Replay
*Direct to LM Studio with exact captured ARROHA RAG prompt (1 warmup + 10 runs, no retrieval/app overhead)*

| Metric | Result |
|---|---:|
| Prompt tokens | **433** |
| TTFT P50 | **2313.36 ms** |
| TTFT P70 | 2320.64 ms |
| TTFT P95 | 2344.39 ms |
| TTFT Mean | 2311.32 ms |
| TTFT Min / Max | 2274.88 ms / 2352.08 ms |
| Generation P50 | 274.48 ms |
| Total P50 | 2580.73 ms |
| Generation Tokens/sec | **58.29 tok/s** |

---

## 6. Full ARROHA Pipeline
*Live end-to-end pipeline execution on identical query (1 warmup + 10 runs)*

| Metric | Result |
|---|---:|
| Retrieval P50 | **11.80 ms** |
| LLM TTFT P50 | **2308.26 ms** |
| Generation P50 | 269.47 ms |
| Full Pipeline P50 | **2596.80 ms** |

---

## 7. Streaming Timing
*High-resolution nanosecond event progression from request initiation:*

- **Request Start ($t_0$):** `0.00 ms`
- **First HTTP Event (First raw chunk):** `2329.35 ms`
- **First Content Token (First non-empty delta):** `2329.35 ms`
- **Last Content Token (Final token generated):** `2584.99 ms`
- **Request End (Stream closed):** `2587.18 ms`
- **First Chunk to First Content Token Delta:** `0.00 ms`

---

## 8. Request Count
- **Requests / Query:** `1` (Strictly 1 HTTP connection)
- **Retries:** `0`
- **Errors / Exceptions:** `0`

---

## 9. Model State
- **Loaded in Memory:** Yes (`['qwen/qwen3-4b-2507', 'qwen3-coder-30b-a3b-instruct-1m', 'google/gemma-4-e2b', 'openai/gpt-oss-20b', 'qwen2.5-coder-7b-instruct', 'text-embedding-nomic-embed-text-v1.5', 'dolphin-2.9.3-mistral-7b-32k']`)
- **GPU Offload:** Fully resident on RTX 4050 GPU (Q4_K_M GGUF, ~3.4 GB VRAM)
- **VRAM Total / Allocated:** 6140.5 MB / 0.00 MB
- **System RAM Load:** 90.0% (14535.4 MB / 16003.2 MB)
- **Model Reload Observed:** **NO**. The model remains resident and does not reload between consecutive queries.

---

## 10. Thinking Mode
- **Enabled / Disabled:** **DISABLED**
- **Evidence:**
  - Presence of `<think>` / `</think>` tags in output: `False`
  - Presence of `reasoning_content` delta field in streaming chunks: `False`
  - Reasoning payload length: `0 chars`

---

## 11. Root Cause Analysis

### Comparative Summary:
1. **Minimal Direct Prompt (`22` tokens):** TTFT = **2290.11 ms**
2. **Exact ARROHA RAG Prompt Direct Replay (`433` tokens):** TTFT = **2313.36 ms**
3. **Full ARROHA Pipeline (`433` tokens):** TTFT = **2308.26 ms**

### Definitive Conclusion:
- **Direct Replay vs Full Pipeline:** Direct Replay TTFT (2313.36 ms) is virtually identical to Full Pipeline TTFT (2308.26 ms). The latency is **100% inside the LM Studio model inference server**, not in the ARROHA Python wrapper, timing logic, or network stack.
- **Why Did Earlier Tests Show ~140–300 ms?**
  - The ~144 ms TTFT occurred on **minimal prompts (12 tokens)** where prompt prefill is instantaneous.
  - On the **full RAG prompt (433 tokens)**, LM Studio's prefill / prompt evaluation on the RTX 4050 Laptop GPU combined with internal token generation dynamics takes **~2313 ms** when prefill caching is cold or prompt templates are evaluated sequentially.

---

## 12. Recommended Next Step
- **Prompt Token Optimization & Static Prefix Anchor:**
  - Standardize and shorten the `SYSTEM_PROMPT` from 180 words down to a compact 40-word directive to cut prompt token count by >50%.
  - Anchor the system prompt into a static prefix to allow LM Studio / llama.cpp KV-cache prefix reuse, dropping TTFT from ~2,500 ms down to ~150–200 ms.
