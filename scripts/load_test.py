"""End-to-end latency and token load test against /api/chat.

Asks in-scope eval questions with the cache off (fresh answers), then
repeats the first few with the cache on (cache hits). Reports client and
server latency, tokens and decisions, and prints the trace_report command
for the per-stage breakdown of exactly these requests.

Usage (inside the container, the API must be running):
    python -m scripts.load_test --limit 100 --seed 7 --cached 25 --label baseline
    python -m scripts.load_test --limit 20 --concurrency 5 --label conc5
"""

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.scope_eval import load_eval
from src.observability.tracing import latency_stats


def _call(base: str, api_key: str, body: dict) -> dict:
    req = urllib.request.Request(
        base + "/api/chat", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "X-API-Key": api_key})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())


def pick_questions(dataset: str, limit: int | None, seed: int | None) -> list[str]:
    """In-scope eval questions, shuffled with seed (if given), first limit."""
    questions = list(dict.fromkeys(load_eval(dataset)[0]))       # unique, in order
    if seed is not None:
        random.Random(seed).shuffle(questions)
    return questions[:limit] if limit else questions


def decision(result: dict) -> str:
    if (result.get("cache") or {}).get("hit"):
        return "cache_hit"
    if (result.get("guardrails") or {}).get("classified_as") == "out_of_scope":
        return "out_of_scope"
    if result.get("abstained"):
        return "abstained"
    return "answered"


def ask(base: str, api_key: str, query: str, use_cache: bool, call=None) -> dict:
    """One request; returns a row with client time, server time, tokens and decision."""
    call = call or _call
    t0 = time.perf_counter()
    try:
        result = call(base, api_key, {"query": query, "use_cache": use_cache})
    except urllib.error.HTTPError as e:
        return {"query": query, "error": f"HTTP {e.code}"}
    except (urllib.error.URLError, TimeoutError) as e:
        return {"query": query, "error": type(e).__name__}
    perf = result.get("perf") or {}
    return {"query": query, "client_ms": round((time.perf_counter() - t0) * 1000, 1),
            "server_ms": perf.get("total_ms"), "tokens": (perf.get("tokens") or {}).get("total", 0),
            "decision": decision(result), "trace_id": result.get("trace_id")}


def run_phase(base, api_key, queries, use_cache, delay, concurrency, call=None) -> list[dict]:
    caches = use_cache if isinstance(use_cache, list) else [use_cache] * len(queries)
    if concurrency <= 1:
        rows = []
        for i, (q, c) in enumerate(zip(queries, caches)):
            if i and delay:
                time.sleep(delay)
            rows.append(ask(base, api_key, q, c, call))
            r = rows[-1]
            print(f"[{i + 1}/{len(queries)}] {r.get('decision', r.get('error')):<12} "
                  f"{r.get('client_ms', 0):>8.0f} ms  {q[:60]}", flush=True)
        return rows
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(lambda qc: ask(base, api_key, qc[0], qc[1], call), zip(queries, caches)))


def summarize(rows: list[dict], expect: str) -> dict:
    """Stats over rows with the expected outcome; others are counted, not timed."""
    ok = [r for r in rows if not r.get("error")]
    timed = [r for r in ok if (r["decision"] == "cache_hit") == (expect == "cache_hit")]
    answered = [r for r in timed if r["decision"] == "answered"]
    return {
        "requests": len(rows),
        "errors": len(rows) - len(ok),
        "decisions": dict(Counter(r["decision"] for r in ok)),
        "unexpected": len(ok) - len(timed),
        "client_ms": latency_stats([r["client_ms"] for r in timed]),
        "server_ms": latency_stats([r["server_ms"] for r in timed if r["server_ms"] is not None]),
        "tokens_per_answer": (round(sum(r["tokens"] for r in answered) / len(answered))
                              if answered else None),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="End-to-end /api/chat load test")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--dataset", default="experiments/eval_dataset.json")
    parser.add_argument("--limit", type=int, default=100, help="fresh questions")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cached", type=int, default=25, help="questions repeated as cache hits")
    parser.add_argument("--delay", type=float, default=0.0, help="seconds between requests")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--label", default="default")
    parser.add_argument("--output-dir", default="experiments")
    args = parser.parse_args(argv)

    api_key = os.environ.get("API_KEY")
    if not api_key:
        parser.error("API_KEY is not set")
    questions = pick_questions(args.dataset, args.limit, args.seed)
    cached = questions[:min(args.cached, len(questions))]
    since = datetime.now(timezone.utc).isoformat()
    print(f"{len(questions)} fresh questions (seed {args.seed}), {len(cached)} cache repeats, "
          f"concurrency {args.concurrency}, label {args.label}\n", flush=True)

    t0 = time.perf_counter()
    # Fresh: cache off, except the questions repeated later, which must be stored first
    fresh = run_phase(args.base_url, api_key, questions,
                      [q in set(cached) for q in questions], args.delay, args.concurrency)
    fresh_s = time.perf_counter() - t0
    hits = run_phase(args.base_url, api_key, cached, True, args.delay, args.concurrency)

    report = {"label": args.label, "started_at": since, "seed": args.seed,
              "concurrency": args.concurrency, "delay_s": args.delay,
              "fresh": {**summarize(fresh, "fresh"), "wall_s": round(fresh_s, 1)},
              "cached": summarize(hits, "cache_hit"), "rows": fresh + hits}
    for name in ("fresh", "cached"):
        s = report[name]
        print(f"\n{name}: {s['requests']} requests, errors {s['errors']}, "
              f"decisions {s['decisions']}, not timed {s['unexpected']}")
        print(f"  client {s['client_ms']}\n  server {s['server_ms']}")
    print(f"  tokens per answered request: {report['fresh']['tokens_per_answer']}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"load_test_{args.label}_{since[:19].replace(':', '-')}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved: {path}")
    print(f"Per-stage breakdown of these requests:\n"
          f"  python -m scripts.trace_report --since {since}")
    return 1 if report["fresh"]["errors"] or report["cached"]["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())