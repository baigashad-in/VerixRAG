"""Measure keyword scope seeding without LLM calls.

For each variant, reports how many in-scope questions skip the LLM classifier
(coverage) and how many off-topic questions are wrongly let through (leakage).

Usage:
    python -m scripts.scope_eval --dataset experiments/eval_dataset.json
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.guardrails.query_classifier import KeywordClassifier
from src.guardrails.scope_terms import alias_terms, build_scope_terms
from src.retrieval.scoping import load_domain_config

# Off-topic questions that share vocabulary with a typical support knowledge base.
PROBES = [
    "What is the capital of France?",
    "Who won the 2022 World Cup?",
    "Recommend a good laptop for programming.",
    "How do I reset my Netflix password?",
    "What's the best email client for Mac?",
    "How do I return a gift on Amazon?",
    "Write Python code to sort a list.",
    "What's a good stock to buy today?",
    "How late is the pharmacy open tonight?",
    "How long does DHL shipping take from China to the US?",
    "Explain how a cloud data center works.",
    "What's your opinion on remote work?",
    "How do I lose weight fast?",
    "Translate 'hello' into Spanish.",
    "What is the speed of light?",
    "Who is the CEO of Apple?",
    "Give me a recipe for pancakes.",
    "How do I file my taxes?",
    "What's the current inflation rate?",
    "Which phone has the best camera?",
]
VARIANTS = [  # (body terms per document, minimum matching terms)
    (0, 1), (10, 1), (0, 2), (10, 2),
]


def load_eval(path: str) -> tuple[list[str], list[str]]:
    raw = json.loads(Path(path).read_text())
    items = raw["questions"] if isinstance(raw, dict) and "questions" in raw else raw
    in_scope, out_scope = [], []
    for q in items:
        meta = q.get("metadata") if isinstance(q.get("metadata"), dict) else {}
        category = q.get("category") or meta.get("category", "")
        (out_scope if category == "out_of_scope" else in_scope).append(q["question"])
    return in_scope, out_scope


def load_adversarial_oos(path: str) -> list[str]:
    if not Path(path).is_file():
        return []
    raw = json.loads(Path(path).read_text())
    return [q["question"] for q in raw.get("questions", [])
            if q.get("expect") == "out_of_scope_or_refuse"]


def run_variant(terms: set[str], strong: set[str], min_matches: int, in_scope: list[str],
                off_topic: list[tuple[str, str]]) -> dict:
    clf = KeywordClassifier(in_scope_terms=list(terms), strong_terms=list(strong),
                            min_matches=min_matches)
    decisions = [clf.classify(q) for q in in_scope]
    skipped = sum(d["in_scope"] is True for d in decisions)
    blocked = [q for q, d in zip(in_scope, decisions) if d["in_scope"] is False]
    leaked = []
    for source, q in off_topic:
        d = clf.classify(q)
        if d["in_scope"] is True:
            leaked.append({"source": source, "question": q, "reason": d["reason"]})
    by_source = {}
    for source, _ in off_topic:
        by_source.setdefault(source, [0, 0])[1] += 1
    for item in leaked:
        by_source[item["source"]][0] += 1
    return {
        "terms": len(terms),
        "coverage": round(skipped / len(in_scope), 3) if in_scope else None,
        "llm_calls_avoided": skipped,
        "in_scope_blocked_by_patterns": blocked,
        "leakage": round(len(leaked) / len(off_topic), 3) if off_topic else None,
        "leakage_by_source": {k: f"{v[0]}/{v[1]}" for k, v in by_source.items()},
        "leaked": leaked,
    }


def main():
    parser = argparse.ArgumentParser(description="Keyword scope seeding evaluation (no LLM calls)")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--documents", default=os.getenv("DOCUMENTS_DIR", "documents"))
    parser.add_argument("--adversarial", default="datasets/adversarial_eval.json")
    parser.add_argument("--domains", default=os.getenv("DOMAINS_CONFIG", "config/domains.json"))
    parser.add_argument("--output-dir", default="experiments")
    args = parser.parse_args()

    aliases = load_domain_config(args.domains).aliases
    strong = alias_terms(aliases)
    in_scope, eval_oos = load_eval(args.dataset)
    off_topic = ([("eval", q) for q in eval_oos]
                 + [("adversarial", q) for q in load_adversarial_oos(args.adversarial)]
                 + [("probes", q) for q in PROBES])
    print(f"{len(in_scope)} in-scope questions, {len(off_topic)} off-topic "
          f"({len(eval_oos)} eval, {len(off_topic) - len(eval_oos) - len(PROBES)} adversarial, "
          f"{len(PROBES)} probes)\n")

    report = {"timestamp": datetime.now().isoformat(), "variants": {}}
    print(f"{'body terms':>10} {'min match':>10} {'terms':>6} {'coverage':>9} "
          f"{'leakage':>8}  leakage by source")
    for top_n, min_matches in VARIANTS:
        terms = build_scope_terms(args.documents, aliases, top_n=top_n)
        result = run_variant(terms, strong, min_matches, in_scope, off_topic)
        result["term_list"] = sorted(terms)
        report["variants"][f"body{top_n}_min{min_matches}"] = result
        print(f"{top_n:>10} {min_matches:>10} {result['terms']:>6} {result['coverage']:>9} "
              f"{result['leakage']:>8}  {result['leakage_by_source']}")

    for name, result in report["variants"].items():
        if result["leaked"]:
            print(f"\nLeaked in {name}:")
            for item in result["leaked"][:12]:
                print(f"  [{item['source']}] {item['question'][:70]}  ({item['reason']})")
        if result["in_scope_blocked_by_patterns"]:
            print(f"In-scope blocked by patterns in {name}: {result['in_scope_blocked_by_patterns'][:5]}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(args.output_dir) / f"scope_eval_{report['timestamp'][:19].replace(':', '-')}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()