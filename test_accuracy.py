"""
test_accuracy.py
----------------
Verify high-precision answer extraction across test queries.
"""

from app.pipeline import RAGPipeline
from app.schemas.query import QueryRequest

pipeline = RAGPipeline()
pipeline.query_cache.clear()

queries = [
    "capital of karnataka",
    "capital of maharashtra",
    "capital of telangana",
    "What is retrieval augmented generation?",
    "capital of gujarat",  # Out of domain -> should trigger refusal
]

for q in queries:
    req = QueryRequest(query=q, language="en")
    resp = pipeline.process_query(req)
    print(f"\nQuery: {q}")
    print(f"Answer: {resp.answer}")
    print(f"Top Source Score: {resp.sources[0].score if resp.sources else 0.0}")
