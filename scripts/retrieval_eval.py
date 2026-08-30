"""
Retrieval metrics (Hit@k, Recall@k, Precision@k, MRR, NDCG) per configuration.

No LLM calls and no quota: only embedding, search and re-ranking run.
Each configuration adds one component, so the table shows what each
one contributes:

    dense         pgvector cosine search only
    bm25          keyword search only
    hybrid        dense + BM25 fused with RRF
    rerank        hybrid -> cross-encoder re-ranking
    full          rerank -> domain scoping   (what production runs)

Usage (inside the container):
    python -m scripts.retrieval_eval --dataset experiments/eval_dataset.json
    python -m scripts.retrieval_eval --dataset experiments/eval_dataset.json --configs hybrid,full
    python -m scripts.retrieval_eval --dataset ... --label before-floor    # name the report

Relevance is judged by (document, section) -> src/evaluation/retrieval_metrics.py.
"""

import argparse
import dataclasses
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from src.evaluation.retrieval_metrics import (
    QuestionResult, aggregate, chunk_sections, grade, parse_labels, question_metrics,
)
from src.observability.tracing import latency_stats
from src.retrieval.confidence import chunk_score, floor_mask

ALL_CONFIGS = ("dense", "bm25", "hybrid", "rerank", "full")
KS = (1, 3, 5, 8)
K_MAX = max(KS)
OUT_OF_SCOPE = {"out_of_scope"}
HEADLINE = ("hit@1", "hit@5", "recall@5", "recall@8", "precision@5", "mrr",
            "ndcg@5", "ndcg@8", "doc_hit@8", "returned", "noise")
FLOOR_RULES = [("off", None, None)] + [(f"margin {m:g}", None, m) for m in (8, 6, 4, 3, 2)] + \
              [(f"floor {f:g}", f, None) for f in (-8, -6, -4, -2, 0)]


def load_questions(path: str, limit: int | None = None) -> list[dict]:
    raw = json.loads(Path(path).read_text())
    items = raw["questions"] if isinstance(raw, dict) and "questions" in raw else raw
    out = []
    for q in items:
        meta = q.get("metadata") if isinstance(q.get("metadata"), dict) else {}
        category = q.get("category") or meta.get("category", "unknown")
        if category in OUT_OF_SCOPE:
            continue
        out.append({
            "question": q["question"],
            "category": category,
            "source_file": q.get("source_file") or meta.get("source_file"),
            "sections": q.get("sections") or meta.get("sections") or [],
        })
    return out[:limit] if limit else out


def corpus_index(store) -> list[tuple[str, list[str]]]:
    """(source, sections covered) of every stored chunk, for ideal rankings."""
    with store.conn.cursor() as cur:
        cur.execute("""SELECT COALESCE(metadata->>'source', metadata->>'filename'),
                              COALESCE(metadata->>'section', ''), content FROM chunks;""")
        return [(r[0], chunk_sections(r[1], r[2])) for r in cur.fetchall()]


def _meta(result) -> dict:
    if isinstance(result, dict):
        return result.get("metadata") or {}
    return getattr(result, "metadata", None) or {}


def _chunk_id(result):
    return result.get("id") if isinstance(result, dict) else getattr(result, "chunk_id", None)


def make_retrievers(engine, store, bm25) -> dict:
    """query -> ranked results, one function per configuration."""
    embed = engine.embedder.embed_text
    scorer = engine.domain_scorer

    def rerank(query):
        engine.domain_scorer = None                  # same engine, scoping switched off
        try:
            return engine._retrieve_uncached(query, K_MAX, True, None)
        finally:
            engine.domain_scorer = scorer

    return {
        "dense": lambda q: store.search(embed(q), top_k=K_MAX),
        "bm25": lambda q: bm25.search(q, top_k=K_MAX),
        "hybrid": lambda q: engine.hybrid.search(q, embed(q), top_k=K_MAX),
        "rerank": rerank,
        "full": lambda q: engine._retrieve_uncached(q, K_MAX, True, None),
    }


def _content(result) -> str:
    return (result.get("content") if isinstance(result, dict)
            else getattr(result, "content", "")) or ""


def evaluate(questions: list[dict], retrieve, corpus: list[tuple[str, list[str]]]) -> tuple[list, list]:
    results, timings = [], []
    for q in questions:
        labels = parse_labels(q["sections"], q["source_file"])
        sources = {lab.source for lab in labels if lab.source}
        if q["source_file"] and q["source_file"] != "cross_document":
            sources.add(q["source_file"])

        # Which labels exist anywhere in the corpus, and the best possible ranking
        corpus_grades, matchable = [], set()
        for src, sections in corpus:
            g, sat = grade(src, sections, labels, sources)
            corpus_grades.append(g)
            matchable |= sat

        t0 = time.perf_counter()
        ranked = retrieve(q["question"])[:K_MAX]
        timings.append((time.perf_counter() - t0) * 1000)

        grades, satisfied, shown = [], [], []
        for res in ranked:
            meta = _meta(res)
            src = meta.get("source") or meta.get("filename")
            g, sat = grade(src, chunk_sections(meta.get("section"), _content(res)), labels, sources)
            grades.append(g)
            satisfied.append(sorted(sat & matchable))
            score = chunk_score(res) if not isinstance(res, dict) else None
            shown.append({"chunk_id": _chunk_id(res), "source": src,
                          "section": meta.get("section", ""), "grade": g,
                          "score": None if score is None else round(float(score), 3),
                          "chars": len(_content(res))})

        results.append(QuestionResult(
            question=q["question"], category=q["category"], sources=sorted(sources),
            labels=[f"{lab.source or q['source_file']}: {lab.section}" for lab in labels],
            matchable_labels=len(matchable), grades=grades, satisfied=[set(s) for s in satisfied],
            ideal=sorted(corpus_grades, reverse=True)[:K_MAX], retrieved=shown,
        ))
    return results, timings


def floor_simulation(results: list) -> dict | None:
    """Replay this run's scores under each floor rule (a floor only removes chunks)."""
    if not results or any(c["score"] is None for r in results for c in r.retrieved):
        return None
    rows = {}
    for name, floor, margin in FLOOR_RULES:
        per_q, chars = [], []
        for r in results:
            keep = floor_mask([c["score"] for c in r.retrieved], floor, margin)
            sub = dataclasses.replace(
                r, grades=[g for g, k in zip(r.grades, keep) if k],
                satisfied=[s for s, k in zip(r.satisfied, keep) if k])
            per_q.append(question_metrics(sub, KS))
            chars.append(sum(c["chars"] for c, k in zip(r.retrieved, keep) if k))
        m = aggregate(per_q)
        rows[name] = {k: m.get(k) for k in ("returned", "noise", "set_precision", "recall@8",
                                             "hit@1", "mrr", "ndcg@8")}
        rows[name]["context_chars"] = round(statistics.mean(chars)) if chars else 0
    return rows


def score_guide(results: list) -> dict | None:
    """Score and gap-to-best percentiles for relevant, same-document and unrelated chunks."""
    groups: dict[str, dict[str, list]] = {}
    for r in results:
        scores = [c["score"] for c in r.retrieved]
        if not scores or any(s is None for s in scores) or not r.section_labelled:
            continue
        top = max(scores)
        for c in r.retrieved:
            kind = {2: "relevant", 1: "same_document", 0: "unrelated"}[c["grade"]]
            g = groups.setdefault(kind, {"score": [], "gap": []})
            g["score"].append(c["score"])
            g["gap"].append(top - c["score"])

    def pct(values, p):
        values = sorted(values)
        return round(values[min(len(values) - 1, int(p / 100 * len(values)))], 2)

    return {kind: {"n": len(g["score"]),
                   "score_p5_p50_p95": [pct(g["score"], p) for p in (5, 50, 95)],
                   "gap_p5_p50_p95": [pct(g["gap"], p) for p in (5, 50, 95)]}
            for kind, g in groups.items()} or None

def main():
    parser = argparse.ArgumentParser(description="Retrieval metrics without LLM calls")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--configs", default=",".join(ALL_CONFIGS))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--label", default="default")
    parser.add_argument("--output-dir", default="experiments")
    parser.add_argument("--sparse-weight", type=float,
                        help="BM25 weight in hybrid fusion (default: HYBRID_SPARSE_WEIGHT)")
    args = parser.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = set(configs) - set(ALL_CONFIGS)
    if unknown:
        parser.error(f"unknown configs {sorted(unknown)}; choose from {ALL_CONFIGS}")

    from src.factory import build_retrieval_stack
    stack = build_retrieval_stack(with_reranker=bool({"rerank", "full"} & set(configs)))
    engine, store, bm25 = stack["engine"], stack["store"], stack["bm25"]
    engine.cache = None                      # measure retrieval, not cache hits
    if args.sparse_weight is not None:
        engine.hybrid.sparse_weight = args.sparse_weight

    questions = load_questions(args.dataset, args.limit)
    corpus = corpus_index(store)
    retrievers = make_retrievers(engine, store, bm25)
    print(f"{len(questions)} in-scope questions, {len(corpus)} chunks, configs: {configs}, "
          f"sparse weight {engine.hybrid.sparse_weight}")

    report = {"label": args.label, "dataset": args.dataset, "timestamp": datetime.now().isoformat(),
              "chunks": len(corpus), "ks": KS, "sparse_weight": engine.hybrid.sparse_weight,
              "configs": {}}
    full_results = None
    for name in configs:
        print(f"  running {name}...", flush=True)
        results, timings = evaluate(questions, retrievers[name], corpus)
        per_q = [question_metrics(r, KS) for r in results]
        report["configs"][name] = {"metrics": aggregate(per_q), "latency_ms": latency_stats(timings)}
        if name == configs[-1]:
            full_results = (results, per_q)

    # Label quality: how many questions could be graded at section level
    results, per_q = full_results
    labelled = [r for r in results if r.section_labelled]
    report["label_quality"] = {
        "questions": len(results),
        "section_labelled": len(labelled),
        "unmatched_label_examples": [
            {"question": r.question, "labels": r.labels} for r in results if not r.section_labelled][:10],
    }

    # Per-category view of the last configuration
    by_cat: dict[str, list] = {}
    for r, m in zip(results, per_q):
        by_cat.setdefault(r.category, []).append(m)
    report["by_category"] = {c: aggregate(ms) for c, ms in sorted(by_cat.items())}
    report["score_guide"] = score_guide(results)
    report["floor_simulation"] = floor_simulation(results)

    # Misses: labelled questions with nothing relevant in the top K_MAX
    report["misses"] = [
        {"question": r.question, "labels": r.labels, "top3": r.retrieved[:3]}
        for r, m in zip(results, per_q) if r.section_labelled and m.get(f"hit@{K_MAX}") == 0.0][:15]

    # ---- print ----
    lq = report["label_quality"]
    print(f"\nSection-level metrics use {lq['section_labelled']}/{lq['questions']} questions "
          f"(the rest have section labels that match no heading; see label_quality)\n")
    header = f"{'config':<8}" + "".join(f"{k:>12}" for k in HEADLINE) + f"{'p50 ms':>9}"
    print(header)
    print("-" * len(header))
    for name, data in report["configs"].items():
        m = data["metrics"]
        cells = "".join(f"{m.get(k) if m.get(k) is not None else '-':>12}" for k in HEADLINE)
        print(f"{name:<8}{cells}{data['latency_ms'].get('p50', '-'):>9}")

    print(f"\nBy category ({configs[-1]}):")
    for cat, m in report["by_category"].items():
        print(f"  {cat:<18} n={m.get('n_doc_hit@8', 0):<4} hit@5={m.get('hit@5')}  "
              f"recall@8={m.get('recall@8')}  mrr={m.get('mrr')}")
    if report["score_guide"]:
          print(f"\nRerank scores by chunk grade ({configs[-1]}): score and gap to best, p5 / p50 / p95")
          for kind, g in report["score_guide"].items():
              print(f"  {kind:<14} n={g['n']:<5} score {g['score_p5_p50_p95']}  gap {g['gap_p5_p50_p95']}")
    if report["floor_simulation"]:
          print(f"\nScore floor simulation ({configs[-1]}, replayed from this run):")
          cols = ("returned", "noise", "set_precision", "recall@8", "hit@1", "mrr", "ndcg@8",
                  "context_chars")
          print(f"  {'rule':<11}" + "".join(f"{c:>14}" for c in cols))
          for name, row in report["floor_simulation"].items():
              print(f"  {name:<11}" + "".join(f"{row[c] if row[c] is not None else '-':>14}" for c in cols))
    if report["misses"]:
        print(f"\nMisses at {K_MAX} ({configs[-1]}, first {len(report['misses'])}):")
        for miss in report["misses"][:5]:
            top = miss["top3"][0] if miss["top3"] else {}
            print(f"  - {miss['question'][:70]}\n      wanted {miss['labels']}\n"
                  f"      got #1 {top.get('source')} › {top.get('section')}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    stamp = report["timestamp"][:19].replace(":", "-")
    path = Path(args.output_dir) / f"retrieval_eval_{args.label}_{stamp}.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nSaved: {path}")
    store.close()


if __name__ == "__main__":
    main()