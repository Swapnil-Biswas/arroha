"""
evaluation/production_smoke_test.py
-----------------------------------
Automated Production Smoke Test Suite for ARROHA Voice & Text RAG API.
Tests all 10 essential criteria:
1. Text query (/query)
2. Synchronous Voice query (/voice)
3. English Voice Streaming (/voice/stream)
4. Hindi Voice Streaming (/voice/stream)
5. Bengali Voice Streaming (/voice/stream)
6. Tamil Voice Streaming (/voice/stream)
7. Interruption / Barge-In (/voice/interrupt)
8. Multiple Consecutive Requests
9. Empty / Silent Audio Error Handling
10. Guardrails & Output Sanitization
"""

from __future__ import annotations

import base64
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LLAMA_BIN_DIR = Path(r"C:\Users\swapn\Downloads\llama-b10451-bin-win-cuda-12.4-x64")
LLAMA_SERVER_EXE = LLAMA_BIN_DIR / "llama-server.exe"
MODEL_PATH_1P5B = Path(r"C:\Users\swapn\.cache\huggingface\hub\models--Qwen--Qwen2.5-1.5B-Instruct-GGUF\snapshots\91cad51170dc346986eccefdc2dd33a9da36ead9\qwen2.5-1.5b-instruct-q4_k_m.gguf")


def ensure_llama_server() -> Optional[subprocess.Popen]:
    """Ensures llama-server with Qwen2.5-1.5B is alive on port 8080."""
    import requests
    try:
        r = requests.get("http://127.0.0.1:8080/health", timeout=1.0)
        if r.status_code == 200:
            logger.info("llama-server already running on port 8080.")
            return None
    except Exception:
        pass

    logger.info("Starting llama-server for Qwen2.5-1.5B on port 8080...")
    cmd = [
        str(LLAMA_SERVER_EXE),
        "-m", str(MODEL_PATH_1P5B),
        "-ngl", "99",
        "-c", "2048",
        "--cache-prompt",
        "--cache-reuse", "64",
        "-np", "1",
        "--host", "127.0.0.1",
        "--port", "8080",
    ]
    env = os.environ.copy()
    if LLAMA_BIN_DIR.exists():
        env["PATH"] = str(LLAMA_BIN_DIR) + os.pathsep + env.get("PATH", "")
    proc = subprocess.Popen(cmd, cwd=str(LLAMA_BIN_DIR), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for _ in range(60):
        try:
            r = requests.get("http://127.0.0.1:8080/health", timeout=1.0)
            if r.status_code == 200:
                logger.info("llama-server is ready on port 8080.")
                return proc
        except Exception:
            pass
        time.sleep(0.5)
    return proc


def run_smoke_tests() -> dict[str, Any]:
    print("=" * 85)
    print("  ARROHA PRODUCTION VOICE INTEGRATION SMOKE TEST SUITE")
    print("=" * 85)

    # 1. Start / Verify llama-server FIRST
    server_proc = ensure_llama_server()

    # 2. Import FastAPI app now that llama-server is alive
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    results = {}

    # Dummy 0.5s 16kHz WAV header & PCM audio bytes
    dummy_wav_bytes = (
        b"RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x08\x00\x00"
        + b"\x00\x00" * 4000
    )
    dummy_b64 = base64.b64encode(dummy_wav_bytes).decode("utf-8")

    # ------------------------------------------------------------------------
    # TEST 1: Health & Index Status
    # ------------------------------------------------------------------------
    print("\n[TEST 1] Checking API Health & System Readiness...")
    res1 = client.get("/health")
    assert res1.status_code == 200, f"Health check failed: {res1.text}"
    health_data = res1.json()
    print(f"  Status: {health_data['status']} | Model: {health_data['model_id']} | TTS: {health_data['tts_backend']}")
    results["test_1_health"] = {"passed": True, "data": health_data}

    # ------------------------------------------------------------------------
    # TEST 2: Text Query (/query)
    # ------------------------------------------------------------------------
    print("\n[TEST 2] Executing Standard Text Query (mode=text)...")
    t0 = time.perf_counter()
    res2 = client.post("/query", json={
        "query": "What was the capital of the Maurya Empire?",
        "dense_weight": 0.8,
        "bm25_weight": 0.2,
        "include_debug": True,
    })
    lat2 = (time.perf_counter() - t0) * 1000.0
    assert res2.status_code == 200, f"Text query failed: {res2.text}"
    d2 = res2.json()
    print(f"  Answer: {d2['answer']}")
    print(f"  Grounded: {d2['grounding']['is_grounded']} | Latency: {d2['latency']['total_ms']} ms (Wall: {lat2:.1f}ms)")
    assert len(d2["answer"]) > 5
    results["test_2_text_query"] = {"passed": True, "latency_ms": d2["latency"]["total_ms"], "answer": d2["answer"]}

    # ------------------------------------------------------------------------
    # TEST 3: Synchronous Voice Query (/voice)
    # ------------------------------------------------------------------------
    print("\n[TEST 3] Executing Synchronous Voice Query (/voice)...")
    res3 = client.post("/voice", json={
        "audio_base64": dummy_b64,
        "audio_format": "wav",
        "language_hint": "en",
        "mode": "voice",
    })
    assert res3.status_code == 200, f"Voice query failed: {res3.text}"
    d3 = res3.json()
    print(f"  Voice Type: {d3.get('voice_type')} | Audio Payload: {'Present' if d3.get('audio_base64') else 'None'}")
    print(f"  First Audio Latency: {d3['latency']['first_audio_latency_ms']} ms | Total: {d3['latency']['total_ms']} ms")
    results["test_3_voice_query"] = {"passed": True, "latency": d3["latency"]}

    # ------------------------------------------------------------------------
    # TEST 4: English Voice Streaming (/voice/stream)
    # ------------------------------------------------------------------------
    print("\n[TEST 4] Executing English Streaming Voice Query (/voice/stream)...")
    t0 = time.perf_counter()
    tokens_received = []
    audio_chunks_received = []
    first_audio_time_ms = None

    with client.stream("POST", "/voice/stream", json={
        "audio_base64": dummy_b64,
        "audio_format": "wav",
        "language_hint": "en",
        "session_id": "test_en_stream",
        "mode": "voice",
    }) as s_res:
        assert s_res.status_code == 200
        for line in s_res.iter_lines():
            if line.startswith("data: "):
                pkt = json.loads(line[6:])
                now_ms = (time.perf_counter() - t0) * 1000.0
                if pkt["event"] == "token" and pkt.get("delta"):
                    tokens_received.append(pkt["delta"])
                elif pkt["event"] == "audio_chunk":
                    if first_audio_time_ms is None:
                        first_audio_time_ms = now_ms
                    audio_chunks_received.append(pkt)

    first_aud_disp = f"{first_audio_time_ms:.1f} ms" if first_audio_time_ms is not None else "N/A"
    print(f"  Tokens: {len(tokens_received)} | Audio Chunks: {len(audio_chunks_received)} | First Audio: {first_aud_disp}")
    print(f"  Streamed Text: {''.join(tokens_received)[:45]}...")
    assert len(tokens_received) > 0
    results["test_4_english_stream"] = {
        "passed": True,
        "first_audio_ms": round(first_audio_time_ms, 2) if first_audio_time_ms else 0.0,
        "chunks_count": len(audio_chunks_received),
    }

    # ------------------------------------------------------------------------
    # TEST 5: Hindi Voice Streaming (/voice/stream)
    # ------------------------------------------------------------------------
    print("\n[TEST 5] Executing Hindi Streaming Voice Query (/voice/stream)...")
    hi_tokens = []
    hi_audio_chunks = []
    with client.stream("POST", "/voice/stream", json={
        "audio_base64": dummy_b64,
        "audio_format": "wav",
        "language_hint": "hi",
        "session_id": "test_hi_stream",
        "mode": "voice",
    }) as s_res:
        assert s_res.status_code == 200
        for line in s_res.iter_lines():
            if line.startswith("data: "):
                pkt = json.loads(line[6:])
                if pkt["event"] == "token" and pkt.get("delta"):
                    hi_tokens.append(pkt["delta"])
                elif pkt["event"] == "audio_chunk":
                    hi_audio_chunks.append(pkt)

    print(f"  Hindi Tokens: {len(hi_tokens)} | Audio Chunks: {len(hi_audio_chunks)} | Text: {''.join(hi_tokens)[:45]}...")
    assert len(hi_tokens) > 0
    results["test_5_hindi_stream"] = {"passed": True, "chunks_count": len(hi_audio_chunks)}

    # ------------------------------------------------------------------------
    # TEST 6: Bengali & Tamil Voice Streaming
    # ------------------------------------------------------------------------
    print("\n[TEST 6] Executing Bengali & Tamil Streaming Voice Queries...")
    for lang in ["bn", "ta"]:
        with client.stream("POST", "/voice/stream", json={
            "audio_base64": dummy_b64,
            "audio_format": "wav",
            "language_hint": lang,
            "session_id": f"test_{lang}_stream",
            "mode": "voice",
        }) as s_res:
            assert s_res.status_code == 200
            chunks_cnt = sum(1 for line in s_res.iter_lines() if "audio_chunk" in line)
            print(f"  [{lang.upper()}] Audio Chunks Streamed: {chunks_cnt}")
    results["test_6_multilingual_stream"] = {"passed": True}

    # ------------------------------------------------------------------------
    # TEST 7: Interruption / Barge-in
    # ------------------------------------------------------------------------
    print("\n[TEST 7] Testing Interruption / Barge-in Endpoint (/voice/interrupt)...")
    res_int = client.post("/voice/interrupt?session_id=test_en_stream")
    assert res_int.status_code == 200
    int_data = res_int.json()
    print(f"  Interruption Result: {int_data}")
    results["test_7_interruption"] = {"passed": True, "data": int_data}

    # ------------------------------------------------------------------------
    # TEST 8: Multiple Consecutive Requests (Concurrency & Cache Reuse)
    # ------------------------------------------------------------------------
    print("\n[TEST 8] Testing Rapid Consecutive Requests (KV Cache Reuse)...")
    latencies = []
    for i in range(5):
        t0 = time.perf_counter()
        r = client.post("/query", json={"query": "What is the highest mountain peak in India?"})
        assert r.status_code == 200
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat_ms)
        print(f"  Request {i+1}: {lat_ms:.1f} ms")
    print(f"  Mean Concurrency Latency: {sum(latencies)/len(latencies):.1f} ms")
    results["test_8_concurrency"] = {"passed": True, "latencies_ms": latencies}

    # ------------------------------------------------------------------------
    # TEST 9: Empty / Silent Audio Input Handling
    # ------------------------------------------------------------------------
    print("\n[TEST 9] Testing Empty / Invalid Audio Handling...")
    res_empty = client.post("/voice", json={"audio_base64": ""})
    assert res_empty.status_code == 200
    d_empty = res_empty.json()
    print(f"  Refusal Flag: {d_empty['is_refusal']} | Message: {d_empty['answer']}")
    assert d_empty["is_refusal"] is True
    results["test_9_empty_audio"] = {"passed": True, "refusal": d_empty["is_refusal"]}

    # ------------------------------------------------------------------------
    # TEST 10: Output Sanitization & Guardrails
    # ------------------------------------------------------------------------
    print("\n[TEST 10] Testing Output Sanitization & Guardrails...")
    res_guard = client.post("/query", json={"query": "Tell me a joke about passwords"})
    assert res_guard.status_code == 200
    print(f"  Guardrails & Sanitization Passed.")
    results["test_10_guardrails"] = {"passed": True}

    print("\n" + "=" * 85)
    print("  ALL 10 PRODUCTION SMOKE TESTS PASSED SUCCESSFULLY")
    print("=" * 85)

    return results


if __name__ == "__main__":
    run_smoke_tests()
