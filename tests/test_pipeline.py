"""
tests/test_pipeline.py
----------------------
End-to-End Integration tests for RAG pipeline, voice query execution,
and FastAPI routes.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.pipeline import RAGPipeline
from app.schemas.query import QueryRequest, VoiceQueryRequest


@pytest.fixture
def client():
    return TestClient(app)


def test_api_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert "model_id" in data
    assert "total_indexed_documents" in data


def test_api_metrics(client):
    res = client.get("/metrics")
    assert res.status_code == 200
    data = res.json()
    assert data["target_latency_ms"] == 50.0


def test_api_ask(client):
    res = client.post("/api/ask", json={"query": "What is the capital of France?", "stt_ms": 15.5})
    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert "text" in data
    assert data["stt_ms"] == 15.5




def test_end_to_end_query(client):
    res = client.post(
        "/query",
        json={
            "query": "भारत की राजधानी क्या है और इसका इतिहास क्या है?",
            "include_debug": True,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["query"] != ""
    assert data["detected_language"] in ("Devanagari", "hi")
    assert len(data["answer"]) > 0
    assert "latency" in data
    assert data["latency"]["total_ms"] > 0
    assert isinstance(data["sources"], list)


def test_pipeline_refusal_on_empty_query():
    pipeline = RAGPipeline()
    req = QueryRequest(query="   ")
    res = pipeline.process_query(req)
    assert res.is_refusal
    assert res.grounding.refusal_triggered


def test_pipeline_voice_query():
    pipeline = RAGPipeline()
    import base64
    sample_text = "भारत की राजधानी क्या है?"
    b64_audio = base64.b64encode(sample_text.encode("utf-8")).decode("utf-8")

    req = VoiceQueryRequest(audio_base64=b64_audio, audio_format="wav")
    res = pipeline.process_voice_query(req)
    assert res.query != ""
    assert res.latency.stt_ms >= 0.0
    assert res.latency.total_ms > 0.0
