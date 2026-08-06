"""
Summarises the request trace log (experiments/traces.jsonl).

Every /api/chat request writes a JSON line with per-stage timings,
memory and retrieval details. This script turns that log into a report:
latency percentiles per stage, where the time goes, memory, token
use, decisions, and the slowest requests.

Usage:
    python -m scripts.trace_report
    python -m scripts.trace_report experiments/traces.jsonl --slowest 10
    python -m scripts.trace_report --since 2026-10-20T18:00   # only newer traces
    python -m scripts.trace_report --trace cc8ad69a852548b8   # one request in full
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.observability.tracing import latency_stats


def load(paths: list[str], since: str | None = None) -> list[dict]:
    traces = []
    for p in paths:
        for path in (Path(p + ".1"), Path(p)):          # rotated file first (older)
            if not path.is_file():
                continue
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    t = json.loads(line)
                except json.JSONDecodeError:
                    continue                               # half-written last line
                if since and t.get("started_at", "") < since:
                    continue
                traces.append(t)
    return traces


def show_trace(t: dict):
    print(f"trace {t['trace_id']}  {t.get('started_at')}  {t.get('status')}  "
          f"{t.get('total_ms')} ms")
    print(f"query: {t.get('attrs', {}).get('query')}")
    print(f"decision: {t.get('attrs', {}).get('decision')}  gate: {t.get('attrs', {}).get('gate')}")
    print("\nspans (offset → duration, RSS):")
    for s in sorted(t.get("spans", []), key=lambda s: s["offset_ms"]):
        indent = "    " if s.get("parent") else "  "
        print(f"{indent}{s['name']:<32} +{s['offset_ms']:>9.1f} ms  {s['duration_ms']:>9.1f} ms  "
              f"{s.get('rss_mb', '?')} MB ({s.get('rss_delta_mb', 0):+} MB)")
    print("\nretrieved:")
    for r in t.get("attrs", {}).get("retrieved", []):
        print(f"  #{r['rank']} {r['source']:<24} id {r['chunk_id']:<5} score {r['rerank_score']}")


def report(traces: list[dict], slowest: int):
    chats = [t for t in traces if t.get("kind") == "chat"] or traces
    print(f"{len(traces)} traces ({len(chats)} chat), "
          f"{chats[0].get('started_at', '?')[:19]} → {chats[-1].get('started_at', '?')[:19]}\n")

    total = latency_stats([t["total_ms"] for t in chats])
    print(f"End-to-end ms: p50 {total.get('p50')}  p95 {total.get('p95')}  max {total.get('max')}")

    stage_values: dict[str, list[float]] = defaultdict(list)
    for t in chats:
        for stage, ms in t.get("stages_ms", {}).items():
            stage_values[stage].append(ms)
        for s in t.get("spans", []):
            if s.get("parent"):                            # sub-stages, e.g. retrieval.rerank
                stage_values[s["name"]].append(s["duration_ms"])
    grand = sum(t["total_ms"] for t in chats) or 1
    top_level = {k for t in chats for k in t.get("stages_ms", {})}

    print(f"\n{'stage':<32}{'n':>5}{'p50':>10}{'p95':>10}{'max':>10}{'share':>8}")
    for stage, vals in sorted(stage_values.items(), key=lambda kv: -sum(kv[1])):
        st = latency_stats(vals)
        share = f"{sum(vals) / grand:.0%}" if stage in top_level else ""
        name = stage if stage in top_level else "  " + stage
        print(f"{name:<32}{st['n']:>5}{st['p50']:>10}{st['p95']:>10}{st['max']:>10}{share:>8}")

    rss = [t["memory_mb"]["rss_end"] for t in chats if t.get("memory_mb")]
    peak = max((t["memory_mb"].get("process_peak", 0) for t in chats if t.get("memory_mb")), default=0)
    if rss:
        print(f"\nMemory: RSS at end of request min {min(rss)} / max {max(rss)} MB, "
              f"process peak {peak} MB")

    tokens = [t["tokens"]["total"] for t in chats if t.get("tokens", {}).get("total")]
    if tokens:
        print(f"Tokens per answered request: mean {sum(tokens) // len(tokens)}, "
              f"total {sum(tokens)}")

    decisions = Counter(t.get("attrs", {}).get("decision") or t.get("status") for t in chats)
    print(f"Decisions: {dict(decisions)}")
    errors = [t for t in chats if t.get("status") == "error"]
    if errors:
        print(f"Errors: {len(errors)} (e.g. trace {errors[-1]['trace_id']})")

    print(f"\nSlowest {slowest}:")
    for t in sorted(chats, key=lambda t: -t["total_ms"])[:slowest]:
        stages = t.get("stages_ms", {})
        worst = max(stages, key=stages.get) if stages else "?"
        q = (t.get("attrs", {}).get("query") or "")[:60]
        print(f"  {t['total_ms']:>9.1f} ms  {t['trace_id']}  worst: {worst} "
              f"{stages.get(worst, 0):.0f} ms  | {q}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", default=["experiments/traces.jsonl"])
    parser.add_argument("--since", help="only traces started at or after this ISO time")
    parser.add_argument("--slowest", type=int, default=5)
    parser.add_argument("--trace", help="print one trace in full")
    args = parser.parse_args()

    traces = load(args.paths, args.since)
    if not traces:
        print("No traces found. Send a few /api/chat requests first.")
        return
    if args.trace:
        match = [t for t in traces if t["trace_id"] == args.trace]
        if not match:
            print(f"Trace {args.trace} not found")
            return
        show_trace(match[0])
        return
    report(traces, args.slowest)


if __name__ == "__main__":
    main()