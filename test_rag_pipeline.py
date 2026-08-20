"""
test_rag_pipeline.py
--------------------
Execute sample query on RAGPipeline and measure latency against <70ms target.
"""

import time
from app.pipeline import RAGPipeline
from app.schemas.query import QueryRequest

print("Initializing RAGPipeline...")
pipeline = RAGPipeline()

query = "What is retrieval augmented generation?"

print(f"\n--- Run 1 (Cold / Uncached Query) ---")
req1 = QueryRequest(query=query, language="en")
resp1 = pipeline.process_query(req1)

print(f"Query: {resp1.query}")
print(f"Answer: {resp1.answer[:100]}...")
print(f"Total Latency: {resp1.latency.total_ms} ms")
print(f"Retrieval Fusion Latency: {resp1.latency.hybrid_fusion_ms} ms")
print(f"Target <70ms Achieved: {resp1.latency.target_achieved_200ms}")

print(f"\n--- Run 2 (Warm / Cached Query) ---")
t0 = time.perf_counter_ns()
resp2 = pipeline.process_query(req1)
t_cached_ms = (time.perf_counter_ns() - t0) / 1e6

print(f"Query: {resp2.query}")
print(f"Answer: {resp2.answer[:100]}...")
print(f"Cached Total Latency: {resp2.latency.total_ms} ms (Measured: {t_cached_ms:.2f} ms)")
print(f"Target <70ms Achieved: {resp2.latency.target_achieved_200ms}")
print(f"Cache Hit: {resp2.debug_info.get('cache_hit') if resp2.debug_info else False}")
