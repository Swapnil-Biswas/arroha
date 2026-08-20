"""
run.py
------
Root CLI entry point for HH Goa 2026 Multilingual Voice RAG:
- Start API Server (FastAPI + Uvicorn)
- Build Indexes (FAISS + BM25)
- Run Latency Benchmark (P50, P70, P100)
- Run Retrieval Evaluation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import uvicorn
from app.config import API_HOST, API_PORT


def main() -> None:
    parser = argparse.ArgumentParser(description="HH Goa 2026 Multilingual Voice RAG CLI.")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI server")
    serve_parser.add_argument("--host", default=API_HOST, help="API host")
    serve_parser.add_argument("--port", type=int, default=API_PORT, help="API port")
    serve_parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")

    # Index command
    index_parser = subparsers.add_parser("index", help="Build FAISS and BM25 indexes")
    index_parser.add_argument("--sample", action="store_true", default=True, help="Build sample dataset")
    index_parser.add_argument("--full", action="store_true", help="Download and index full language shards")
    index_parser.add_argument("--strategy", default="sentence", choices=["fixed", "sentence", "passage", "recursive"])

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run latency & retrieval benchmark")
    bench_parser.add_argument("--num-queries", type=int, default=20, help="Number of benchmark query runs")

    args = parser.parse_args()

    if args.command == "serve" or args.command is None:
        host = getattr(args, "host", API_HOST)
        port = getattr(args, "port", API_PORT)
        reload = getattr(args, "reload", False)
        print(f"Starting HH Goa Voice RAG Server on http://{host}:{port}...")
        if reload:
            uvicorn.run("app.main:app", host=host, port=port, reload=True)
        else:
            from app.main import app as fastapi_app
            uvicorn.run(fastapi_app, host=host, port=port)

    elif args.command == "index":
        from ingestion.build_index import build_pipeline_indexes
        use_sample = not args.full
        build_pipeline_indexes(use_sample=use_sample, chunking_strategy=args.strategy)

    elif args.command == "benchmark":
        from evaluation.latency import run_latency_benchmark
        run_latency_benchmark(num_runs=args.num_queries)


if __name__ == "__main__":
    main()
