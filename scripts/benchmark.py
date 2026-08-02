"""
Latency and memory benchmark for the retrieval stack.

Retrieval-only mode. It answers:
  - how long does each stage take? (embed, dense, sparse, rerank etc.)
  - what are p50 / p95 / p99 latencies?
  - how much memory do the models and indexes use?

--full also runs generation through the RAG pipeline (uses Groq quota).

Usage (inside the container):
    python -m scripts.benchmark
    python -m scripts.benchmark --dataset experiments/eval_subset_45.json --repeat 3
    python -m scripts.benchmark --full --limit 5
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.factory import build_retrieval_stack
from src.observability.tracing import (
    TraceStore, latency_stats, peak_rss_mb, rss_mb, set_trace_store, start_trace,
)

DEFAULT_QUERIES = [
    "What are the password requirements for creating an account?",
    "How do I enable two-factor authentication?",
    "How long is a password reset link valid?",
    "What happens to my store credit if I delete my account?",
    "How much does express shipping cost?",
    "What does error code E-4012 mean?",
    "What is the API rate limit on the Business plan?",
    "How many days of PTO do new employees get?",
]


def load_queries(path: str | None, limit: int | None) -> list[str]:
    if not path:
        queries = DEFAULT_QUERIES
    else:
        raw = json.loads(Path(path).read_text())
        items = raw["questions"] if isinstance(raw, dict) and "questions" in raw else raw
        queries = [q["question"] for q in items]
    return queries[:limit] if limit else queries


def main():
    parser = argparse.ArgumentParser(description="Benchmark VerixRAG latency and memory")
    parser.add_argument("--dataset", help="eval dataset JSON to take questions from")
    parser.add_argument("--limit", type=int, help="max number of questions")
    parser.add_argument("--repeat", type=int, default=1, help="passes over the questions")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--full", action="store_true",
                        help="also run LLM generation (uses API quota)")
    parser.add_argument("--no-cache", action="store_true",
                        help="disable the retrieval cache for this run")
    parser.add_argument("--output", default="experiments/benchmark.json")
    args = parser.parse_args()

    if args.no_cache:
        os.environ["CACHE_ENABLED"] = "false"

    store = TraceStore(maxlen=100_000, log_path="")
    set_trace_store(store)

    stack = build_retrieval_stack()
    engine = stack["engine"]
    queries = load_queries(args.dataset, args.limit)

    pipeline = None
    if args.full:
        from src.query.query_transformer import QueryTransformer
        from src.query.enhanced_retrieval import EnhancedRetriever
        from src.generation.generator import RAGGenerator
        from src.pipeline import RAGPipeline
        enhanced = EnhancedRetriever(engine, QueryTransformer(), stack["embedder"])
        pipeline = RAGPipeline(enhanced, RAGGenerator())

    # Warm-up: first call pays one-off costs (lazy init, allocator growth)
    engine.retrieve(queries[0], top_k=args.top_k)

    print(f"Benchmarking {len(queries)} queries x {args.repeat} "
          f"({'full pipeline' if args.full else 'retrieval only'})")
    wall_start = time.perf_counter()
    for _ in range(args.repeat):
        for q in queries:
            if pipeline:
                pipeline.answer(q, top_k=args.top_k, check_hallucination=False)
            else:
                with start_trace("benchmark", store=store):
                    engine.retrieve(q, top_k=args.top_k)
    wall_s = time.perf_counter() - wall_start

    metrics = store.metrics()
    n = len(queries) * args.repeat
    report = {
        "timestamp": datetime.now().isoformat(),
        "mode": "full" if args.full else "retrieval_only",
        "queries": n,
        "throughput_qps": round(n / wall_s, 2) if wall_s else None,
        "latency_ms": metrics["latency_ms"],
        "stage_latency_ms": metrics["stage_latency_ms"],
        "tokens_per_request": metrics["tokens_per_request"],
        "memory_mb": {
            "startup": stack["startup_perf"],
            "rss_end": rss_mb(),
            "process_peak": peak_rss_mb(),
            "rss_delta_per_request": metrics["memory_mb"]["rss_delta_per_request"],
        },
        "caches": engine.cache_stats() if hasattr(engine, "cache_stats") else None,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))

    lat = report["latency_ms"]
    print(f"\nThroughput: {report['throughput_qps']} queries/s")
    print(f"Latency: p50 {lat.get('p50')} ms | p95 {lat.get('p95')} ms | "
          f"p99 {lat.get('p99')} ms | max {lat.get('max')} ms")
    print("Per stage (p50 / p95 ms):")
    for name, s in report["stage_latency_ms"].items():
        print(f"  {name:<28} {s.get('p50')} / {s.get('p95')}")
    print("Model load (ms, RSS delta MB):")
    for name, v in stack["startup_perf"].items():
        if isinstance(v, dict):
            print(f"  {name:<28} {v['ms']} ms, +{v['rss_delta_mb']} MB")
    print(f"Memory: RSS {report['memory_mb']['rss_end']} MB, "
          f"peak {report['memory_mb']['process_peak']} MB")
    print(f"Saved: {args.output}")
    stack["store"].close()


if __name__ == "__main__":
    main()