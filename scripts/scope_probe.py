"""End-to-end off-topic probe: which layer stops each off-topic question. """

import argparse
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.scope_eval import PROBES, load_adversarial_oos, load_eval
from src.evaluation.adversarial import GAP, OUT_OF_SCOPE, REFUSAL, normalize_answer

LAYERS = ["keyword", "llm", "gate", "generator", "partial", "answered"]
_CITED = re.compile(r"[\[(]\s*sources?\s*\d", re.I)


def _call(base: str, path: str, api_key: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(
        base + path, data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json", "X-API-Key": api_key})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def classify_attrs(trace: dict) -> dict:
    """Attributes of the guardrails.classify span, or {} if missing."""
    for s in trace.get("spans", []):
        if s.get("name") == "guardrails.classify":
            return s.get("attrs", {})
    return {}


def classifier_path(attrs: dict) -> str:
    stage = attrs.get("stage") or "unknown"
    return "llm_failed_open" if stage == "llm" and attrs.get("fallback") else stage


def stopped_by(result: dict) -> str:
    """The layer that handled an off-topic question; 'answered' is a leak."""
    guard = result.get("guardrails") or {}
    if guard.get("classified_as") == "out_of_scope":
        return "llm" if guard.get("stage") == "llm" else "keyword"
    if result.get("abstained"):
        return "gate"
    text = normalize_answer(result.get("answer", ""))
    if REFUSAL.search(text) or OUT_OF_SCOPE.search(text):
        return "generator"
    if GAP.search(text):
        return "partial" if _CITED.search(text) else "generator"
    return "answered"


def load_questions(dataset: str, adversarial: str) -> list[tuple[str, str]]:
    seen, out = set(), []
    sources = ([("probe", q) for q in PROBES]
               + [("eval", q) for q in (load_eval(dataset)[1] if Path(dataset).is_file() else [])]
               + [("adversarial", q) for q in load_adversarial_oos(adversarial)])
    for source, q in sources:
        if q not in seen:
            seen.add(q)
            out.append((source, q))
    return out


def summarize(rows: list[dict]) -> dict:
    scored = [r for r in rows if not r.get("error")]
    by_source = {}
    for r in scored:
        by_source.setdefault(r["source"], Counter())[r["stopped_by"]] += 1
    ms = [r["ms"] for r in scored]
    return {
        "questions": len(rows),
        "errors": len(rows) - len(scored),
        "stopped_by": {k: sum(r["stopped_by"] == k for r in scored) for k in LAYERS},
        "by_source": {s: dict(c) for s, c in by_source.items()},
        "classifier_path": dict(Counter(r["classifier"] for r in scored)),
        "leaks": sum(r["stopped_by"] == "answered" for r in scored),
        "latency_ms": {"p50": round(statistics.median(ms), 1) if ms else None,
                       "p95": (round(statistics.quantiles(ms, n=20)[-1], 1)
                               if len(ms) >= 2 else None)},
    }


def main():
    parser = argparse.ArgumentParser(description="End-to-end off-topic probe")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--dataset", default="experiments/eval_dataset.json")
    parser.add_argument("--adversarial", default="datasets/adversarial_eval.json")
    parser.add_argument("--delay", type=float, default=2.0, help="seconds between requests")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--label", default="default")
    parser.add_argument("--output-dir", default="experiments")
    args = parser.parse_args()

    api_key = os.environ.get("API_KEY")
    if not api_key:
        parser.error("API_KEY is not set")
    questions = load_questions(args.dataset, args.adversarial)[:args.limit]
    print(f"{len(questions)} off-topic questions, label {args.label}\n")

    rows = []
    for i, (source, q) in enumerate(questions, 1):
        if i > 1 and args.delay:
            time.sleep(args.delay)
        t0 = time.perf_counter()
        try:
            result = _call(args.base_url, "/api/chat", api_key, {"query": q, "use_cache": False})
        except urllib.error.HTTPError as e:
            rows.append({"source": source, "question": q, "error": f"HTTP {e.code}"})
            print(f"[{i}/{len(questions)}] ERROR HTTP {e.code}: {q[:60]}", flush=True)
            continue
        ms = round((time.perf_counter() - t0) * 1000, 1)
        try:
            trace = _call(args.base_url, f"/api/traces/{result.get('trace_id')}", api_key)
        except urllib.error.HTTPError:
            trace = {}
        attrs = classify_attrs(trace)
        row = {"source": source, "question": q, "stopped_by": stopped_by(result),
               "classifier": classifier_path(attrs), "fallback": attrs.get("fallback"),
               "ms": ms, "answer": normalize_answer(result.get("answer", ""))[:400]}
        rows.append(row)
        print(f"[{i}/{len(questions)}] {row['stopped_by']:<9} via {row['classifier']:<15} "
              f"{ms:>7.0f} ms  {q[:60]}", flush=True)

    summary = summarize(rows)
    print(f"\nStopped by: {summary['stopped_by']}  (errors {summary['errors']})")
    for source, counts in summary["by_source"].items():
        print(f"  {source:<12} {counts}")
    print(f"Classifier path: {summary['classifier_path']}")
    print(f"Client latency: {summary['latency_ms']}")
    print(f"Leaks (answered): {summary['leaks']}")
    for r in rows:
        if r.get("stopped_by") in ("answered", "partial"):
            print(f"  [{r['stopped_by']}] {r['question']}\n      {r['answer'][:200]}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(args.output_dir) / (f"scope_probe_{args.label}_"
                                    f"{datetime.now():%Y%m%d_%H%M%S}.json")
    path.write_text(json.dumps({"label": args.label, "timestamp": datetime.now().isoformat(),
                                "summary": summary, "results": rows}, indent=2))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()