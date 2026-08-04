"""
Calibrate the confidence gate from real retrieval scores.

For every question in an eval dataset, run retrieval + re-ranking and
record the best cross-encoder score. Then, for a range of thresholds,
show how many in-scope questions the gate would wrongly block and how
many out-of-scope questions it would catch.

Run this script whenever documents or models change.

Usage:
    python -m scripts.calibrate --dataset experiments/eval_dataset.json
    python -m scripts.calibrate --dataset experiments/eval_dataset.json --max-in-scope-loss 0.01

Set RERANK_MIN_SCORE in .env to the recommended value (or a bit
lower, to keep a safety margin) and rebuild.

Caution: Calibrating on the same questions used to report evaluation scores 
may overfit the gate to that set. Use a conservative value.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.observability.tracing import latency_stats

OUT_OF_SCOPE = {"out_of_scope"}


def load_items(path: str, limit: int | None) -> list[dict]:
    raw = json.loads(Path(path).read_text())
    items = raw["questions"] if isinstance(raw, dict) and "questions" in raw else raw
    out = []
    for q in items:
        meta = q.get("metadata", {}) if isinstance(q.get("metadata"), dict) else {}
        out.append({
            "question": q["question"],
            "category": q.get("category") or meta.get("category", "unknown"),
            "source_file": q.get("source_file") or meta.get("source_file"),
        })
    return out[:limit] if limit else out


def threshold_table(in_scores, out_scores, lo=-11.0, hi=6.0, step=0.5) -> list[dict]:
    rows = []
    t = lo
    while t <= hi + 1e-9:
        rows.append({
            "threshold": round(t, 2),
            "in_scope_blocked": sum(s < t for s in in_scores),
            "in_scope_blocked_rate": round(sum(s < t for s in in_scores) / len(in_scores), 4) if in_scores else None,
            "out_of_scope_caught": sum(s < t for s in out_scores),
            "out_of_scope_caught_rate": round(sum(s < t for s in out_scores) / len(out_scores), 4) if out_scores else None,
        })
        t += step
    return rows


def recommend(rows, max_loss: float) -> dict | None:
    ok = [r for r in rows if r["in_scope_blocked_rate"] is not None
          and r["in_scope_blocked_rate"] <= max_loss]
    return ok[-1] if ok else None


def summarize(scores: list[float]) -> dict:
    if not scores:
        return {"n": 0}
    s = sorted(scores)
    return {**latency_stats(s), "min": round(s[0], 3), "p5": round(s[max(0, int(0.05 * len(s)) - 1)], 3)}


def main():
    parser = argparse.ArgumentParser(description="Calibrate RERANK_MIN_SCORE (no LLM calls)")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-in-scope-loss", type=float, default=0.02,
                        help="max fraction of in-scope questions the gate may block")
    parser.add_argument("--output", default="experiments/calibration.json")
    args = parser.parse_args()

    os.environ["CACHE_ENABLED"] = "false"          # measure real retrieval
    from src.factory import build_retrieval_stack
    stack = build_retrieval_stack()
    engine = stack["engine"]

    items = load_items(args.dataset, args.limit)
    print(f"Scoring {len(items)} questions (retrieval only)...")
    per_question = []
    for i, item in enumerate(items, 1):
        results = engine.retrieve(item["question"], top_k=args.top_k)
        scores = [r.rerank_score for r in results if getattr(r, "rerank_score", None) is not None]
        top = max(scores) if scores else None
        per_question.append({**item, "top_score": round(top, 4) if top is not None else None,
                             "top_source": (results[0].metadata.get("source") if results else None)})
        if i % 25 == 0:
            print(f"  {i}/{len(items)}")

    in_scores = [q["top_score"] for q in per_question
                 if q["top_score"] is not None and q["category"] not in OUT_OF_SCOPE]
    out_scores = [q["top_score"] for q in per_question
                  if q["top_score"] is not None and q["category"] in OUT_OF_SCOPE]
    rows = threshold_table(in_scores, out_scores)
    rec = recommend(rows, args.max_in_scope_loss)

    by_category: dict[str, list[float]] = {}
    for q in per_question:
        if q["top_score"] is not None:
            by_category.setdefault(q["category"], []).append(q["top_score"])

    report = {
        "timestamp": datetime.now().isoformat(),
        "dataset": args.dataset,
        "questions": len(per_question),
        "top_score_by_group": {"in_scope": summarize(in_scores), "out_of_scope": summarize(out_scores)},
        "top_score_by_category": {k: summarize(v) for k, v in sorted(by_category.items())},
        "threshold_table": rows,
        "recommended": rec,
        "max_in_scope_loss": args.max_in_scope_loss,
        "lowest_in_scope_questions": sorted(
            [q for q in per_question if q["category"] not in OUT_OF_SCOPE and q["top_score"] is not None],
            key=lambda q: q["top_score"])[:10],
        "highest_out_of_scope_questions": sorted(
            [q for q in per_question if q["category"] in OUT_OF_SCOPE and q["top_score"] is not None],
            key=lambda q: -q["top_score"])[:10],
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))

    print("\nBest rerank score per question:")
    for group, stats in report["top_score_by_group"].items():
        print(f"  {group:<13} n={stats.get('n')}  min={stats.get('min')}  "
              f"p50={stats.get('p50')}  max={stats.get('max')}")
    print("\nthreshold | in-scope blocked | out-of-scope caught")
    for r in rows:
        if r["in_scope_blocked"] or r["out_of_scope_caught"]:
            print(f"  {r['threshold']:>6} | {r['in_scope_blocked']:>4} ({r['in_scope_blocked_rate']}) "
                  f"| {r['out_of_scope_caught']:>4} ({r['out_of_scope_caught_rate']})")
    if rec:
        print(f"\nRecommended RERANK_MIN_SCORE={rec['threshold']} "
              f"(blocks {rec['in_scope_blocked']} in-scope, catches "
              f"{rec['out_of_scope_caught']} out-of-scope)")
    else:
        print("\nNo threshold meets the in-scope loss limit; keep the gate conservative.")
    print("Lowest-scoring in-scope questions (check these are genuinely unanswerable):")
    for q in report["lowest_in_scope_questions"][:5]:
        print(f"  {q['top_score']:>7}  {q['question'][:80]}")
    print(f"Saved: {args.output}")
    stack["store"].close()


if __name__ == "__main__":
    main()