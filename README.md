# ARROHA — Multilingual Voice-Enabled RAG System (Task 2)
### Hackathon Goa 2026 — AI-Driven Retrieval & Real-Time Spoken Query Pipeline

ARROHA is a high-performance, real-time, multilingual Retrieval-Augmented Generation (RAG) system built over the **ai4bharat/MSMARCO-XI** dataset (50,400 chunks across 15 Indic and global languages). Optimized for spoken dialogue, ARROHA delivers low-latency conversational audio through a concurrent streaming pipeline designed for the **<200 ms post-STT latency requirement**.

---

## 1. System Architecture

```mermaid
flowchart TD
    A[User Voice Input / Spoken Audio] --> B[Speech-to-Text Engine\nWhisper / High-Speed STT]
    B --> C[User Query & Detected Language]
    C --> D[Input Guardrails & Sanitization]
    
    D --> E1[Multilingual Embedder\nparaphrase-multilingual-MiniLM-L12-v2]
    D --> E2[Multilingual Tokenizer]
    
    E1 --> F1[Dense Vector Search\nFAISS IndexFlatIP (50.4k chunks)]
    E2 --> F2[Sparse Lexical Search\nSQLite FTS5 BM25]
    
    F1 --> G[Candidate Fusion & Score Normalization\nDense: 0.8 + BM25: 0.2]
    F2 --> G
    
    G --> H[Top-K Grounded Source Context]
    H --> I[Grounded Prompt Assembly]
    
    I --> J[Qwen2.5-1.5B-Instruct Q4_K_M\nllama-server CUDA 12.4 Streaming]
    
    J -->|Delta Tokens| K[Adaptive Streaming Text Buffer\nBPE Leading-Space Boundaries]
    K -->|Chunk 1 Eager (3-4 words)| L[Local ONNX Streaming Synthesizer\nSub-16ms Acoustic Synthesis]
    K -->|Subsequent Clauses| L
    
    L -->|16-bit 24kHz PCM Chunks| M[Web Audio Queue & Streaming Playback]
    M --> N[Real-Time Conversational Audio Stream]
```

---

## 2. Real-Time Streaming Voice RAG

Traditional voice RAG systems suffer from a sequential **generate-then-speak** latency barrier (often >1,200 ms). ARROHA overcomes this with a **concurrent producer-consumer streaming architecture**:

```
[User Speech] 
      │
      ▼
[STT Layer] 
      │ Transcribed Text
      ▼
[50,400-Chunk Hybrid Retrieval] (FAISS + SQLite FTS5 ~18 ms)
      │ Top-5 Grounded Sources
      ▼
[Qwen2.5-1.5B Streaming Generation] (llama-server CUDA ~103 ms TTFT)
      │ Incremental Delta Tokens
      ▼
[Adaptive Text Buffering] (Eager Chunk 1 at 3-4 words / clause boundary)
      │ Speech-Ready Word Chunks
      ▼
[Local ONNX Streaming Synthesizer] (~15.8 ms synthesis latency)
      │ 24kHz PCM Audio Frames
      ▼
[Streaming Browser Audio Queue] (Instant playback while LLM continues generating)
```

Because Chunk 1 provides ~2.5–4.0 seconds of spoken duration, and Qwen2.5-1.5B completes its entire 24-token response in ~214 ms, the playback queue never starves, yielding **100% audio continuity with 0 starvation gaps**.

### Production API Endpoints

- **`POST /voice/stream`**: Server-Sent Events (SSE) streaming endpoint yielding real-time `status`, `transcript`, delta `token`, synthesized `audio_chunk` frames, and `done` latency metrics.
- **`POST /voice/interrupt`**: Instant barge-in cancellation endpoint that halts active speech synthesis, drains server queues, and returns control to the user.
- **`POST /voice`**: Synchronous voice query endpoint returning full text answer, grounded citations, and base64 WAV payload.
- **`POST /query`**: Text-mode compatibility endpoint for standard non-streaming RAG queries.
- **`GET /health` & `GET /metrics`**: Service health, active index counts, model bindings, and latency instrumentation.

### Production Voice Modules (`app/voice/`)

- **`app/voice/language_router.py`**: Canonical 15-language router mapping native neural voices (`en`, `hi`, `bn`, `ta`, `te`, `mr`, `gu`, `kn`, `ml`, `pa`, `ne`, `ur`) and phonetic fallbacks (`or`, `as`, `sa`).
- **`app/voice/tts_backend.py`**: High-performance local acoustic synthesizer using ONNX Runtime with sub-16ms latency, zero GPU VRAM contention, and cloud Edge-TTS fallback.
- **`app/voice/streaming_buffer.py`**: Deterministic BPE leading-space boundary buffer with eager Chunk 1 emission.
- **`app/voice/pipeline.py`**: Multi-threaded concurrent streaming voice orchestrator with thread-safe session cancellation.
- **`app/voice/stt.py`**: Speech-to-Text engine supporting Whisper and high-speed audio processors.

---

## 3. Validated Production Benchmarks

### Test Environment
- **Hardware:** ASUS ROG Strix G16 (Intel Core i7-13650HX, NVIDIA GeForce RTX 4050 Laptop GPU 6GB GDDR6, 16GB RAM)
- **LLM Backend:** Qwen2.5-1.5B-Instruct Q4_K_M on `llama-server` b10451 (CUDA 12.4, `-ngl 99`, `--cache-reuse 64`, `max_tokens=24`, `temperature=0.1`)
- **Retrieval Corpus:** 50,400 chunks, FAISS `IndexFlatIP` (384-d MiniLM) + SQLite FTS5 lexical index, dense-heavy hybrid fusion (0.8 dense / 0.2 BM25)
- **Benchmark Suite:** 45 canonical multilingual queries across all 15 supported locales

### Live Production Streaming Voice Results

| Metric | Result |
|---|---:|
| **TTFA P50** (Time-to-First-Audio) | **141.94 ms** |
| **TTFA P90** | **179.42 ms** |
| **TTFA P95** | **197.86 ms** |
| **< 200 ms queries** | **95.56% (43 / 45)** |
| **< 150 ms queries** | **66.67% (30 / 45)** |
| **Pre-completion speech** | **100% (45 / 45)** |
| **Audio starvation** | **0 gaps (100% continuity)** |
| **Factual grounding accuracy** | **73.33%** |
| **Production smoke tests** | **10 / 10 (100% passed)** |

> **Performance Summary:** ARROHA achieves a production-validated sub-200 ms real-time voice response target for **95.56% of the 45-query multilingual benchmark**, with a **141.94 ms TTFA P50**. Speech playback begins while the LLM is still generating remaining tokens, enabling instantaneous conversational responsiveness across all 15 Indian and global languages.

---

## 4. Multilingual Dataset & Data Safety

- **Dataset:** `ai4bharat/MSMARCO-XI` (15 languages: Assamese, Bengali, English, Gujarati, Hindi, Kannada, Malayalam, Marathi, Nepali, Odia, Punjabi, Sanskrit, Tamil, Telugu, Urdu).
- **Strict Data-Safety Enforcement:**
  - `Answer` and `Eng_Answer` are **NEVER** placed into the searchable corpus text.
  - `is_selected` is strictly isolated as evaluation metadata.
  - Documents are constructed purely from genuine passages (`Translated_passages` and `English_passages`).

---

## 5. Project Structure

```text
hhgoaRAG/
├── app/
│   ├── main.py                  # FastAPI server with /voice/stream, /voice/interrupt, /query
│   ├── config.py                # Centralized environment configuration
│   ├── pipeline.py              # End-to-end RAG orchestrator with streaming voice support
│   ├── voice/
│   │   ├── language_router.py   # 15-language voice router (Native + Fallback)
│   │   ├── tts_backend.py       # Local ONNX streaming synthesizer (<16ms)
│   │   ├── streaming_buffer.py  # Adaptive BPE boundary token buffer
│   │   ├── pipeline.py          # Concurrent LLM + TTS streaming pipeline
│   │   └── stt.py               # Speech-to-Text layer
│   ├── retrieval/
│   │   ├── bm25.py              # SQLite FTS5 / BM25 lexical retriever
│   │   ├── vector.py            # FAISS dense vector retriever
│   │   ├── hybrid.py            # Hybrid score fusion (Dense 0.8 + BM25 0.2)
│   │   └── reranker.py          # Lightweight optional reranker
│   ├── generation/
│   │   ├── llm.py               # LLM generator with streaming OpenAI-compatible client
│   │   └── prompts.py           # Concise grounded multilingual system prompts
│   ├── guardrails/
│   │   ├── input.py             # Query sanitization, script detection & injection filter
│   │   ├── grounding.py         # Hallucination detection & refusal check
│   │   ├── output.py            # Output length & format bounding
│   │   └── validator.py         # Unified guardrail service
│   ├── schemas/
│   │   ├── query.py             # Pydantic request models (VoiceQueryRequest, QueryRequest)
│   │   └── response.py          # Pydantic models (VoiceStreamChunk, RAGResponse)
│   └── static/                  # Web Demo UI (HTML5, Vanilla CSS, Web Audio JS)
│
├── ingestion/
│   ├── inspect_dataset.py       # Dataset schema inspection tool
│   ├── download.py              # Multilingual shard downloader & corpus generator
│   ├── preprocess.py            # Unicode NFC normalization & document extraction
│   ├── models.py                # Canonical Document, Chunk, and Record models
│   ├── chunking.py              # 4 chunking strategies (sentence, fixed, passage, recursive)
│   └── build_index.py           # Index construction CLI & pipeline
│
├── indexing/
│   ├── embeddings.py            # sentence-transformers multilingual embedder (GPU/CPU)
│   ├── faiss_index.py           # FAISS IndexFlatIP vector index manager
│   └── bm25_index.py            # Multilingual lexical index manager
│
├── evaluation/
│   ├── production_smoke_test.py      # 10-test production validation test suite
│   ├── production_voice_benchmark.py  # 45-query live streaming voice benchmark suite
│   ├── voice_end_to_end_benchmark.py # End-to-end voice streaming condition evaluations
│   ├── model_quality_forensics.py    # Forensic 3-model quality & hallucination benchmark
│   └── results/                      # Validated benchmark reports & raw telemetry (JSON/MD)
│
├── tests/                       # Unit and integration test suite
├── data/                        # Raw & processed corpus data (git-ignored)
├── indexes/                     # Saved FAISS & FTS5 binary indexes (git-ignored)
├── run.py                       # Root launcher CLI
└── requirements.txt
```

---

## 6. Quick Start & Execution

### 1. Environment Setup
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start Inference Server (llama-server)
```powershell
# Run Qwen2.5-1.5B with GPU acceleration on port 8080
llama-server.exe -m path\to\qwen2.5-1.5b-instruct-q4_k_m.gguf -ngl 99 -c 2048 --cache-prompt --cache-reuse 64 --port 8080
```

### 3. Run Production Smoke Tests
```powershell
.venv\Scripts\python evaluation\production_smoke_test.py
```

### 4. Run Live Streaming Voice Benchmark
```powershell
.venv\Scripts\python evaluation\production_voice_benchmark.py
```

### 5. Start the Web Application
```powershell
.venv\Scripts\python run.py serve
# Open browser at: http://localhost:8000
```
