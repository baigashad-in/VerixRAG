"""Compare the old and new uncited-claim rules on saved answers, without LLM calls.

Usage:
    python -m scripts.citation_audit
    python -m scripts.citation_audit experiments/adv_part7.json experiments/ragas_*.json
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.adversarial import normalize_answer
from src.generation.hallucination_check import MIN_CLAIM_WORDS, HallucinationDetector

OLD = {"min_claim_words": 8, "short_facts": False}
NEW = {"min_claim_words": MIN_CLAIM_WORDS, "short_facts": True}


def find_answers(node, found: dict):
    """Collect answer strings from any nested report structure, de-duplicated."""
    if isinstance(node, dict):
        answer = node.get("answer")
        if isinstance(answer, str) and answer.strip():
            text = normalize_answer(answer).strip()
            found.setdefault(text, node.get("question") or node.get("query") or "")
        for value in node.values():
            find_answers(value, found)
    elif isinstance(node, list):
        for value in node:
            find_answers(value, found)


def main():
    parser = argparse.ArgumentParser(description="Audit uncited-claim detection on saved answers")
    parser.add_argument("paths", nargs="*", default=sorted(glob.glob("experiments/*.json")))
    parser.add_argument("--show", type=int, default=15)
    args = parser.parse_args()

    answers: dict[str, str] = {}
    for path in args.paths:
        try:
            find_answers(json.load(open(path)), answers)
        except (OSError, json.JSONDecodeError):
            continue
    if not answers:
        print("No saved answers found.")
        return

    detector = HallucinationDetector()
    totals = {"old": [0, 0, 0.0], "new": [0, 0, 0.0]}      # suspicious, uncited, coverage sum
    newly_uncited = []
    for answer, question in answers.items():
        old = detector.check_citation_coverage(answer, num_sources=8, **OLD)
        new = detector.check_citation_coverage(answer, num_sources=8, **NEW)
        for name, r in (("old", old), ("new", new)):
            totals[name][0] += r["is_suspicious"]
            totals[name][1] += len(r["uncited_claims"])
            totals[name][2] += r["citation_coverage"]
        for sentence in set(new["uncited_claims"]) - set(old["uncited_claims"]):
            newly_uncited.append((question, sentence))

    n = len(answers)
    print(f"{n} unique saved answers from {len(args.paths)} files\n")
    print(f"{'rule':<34}{'suspicious':>12}{'uncited sentences':>19}{'mean coverage':>15}")
    for name, label in (("old", "old (<8 words ignored)"),
                        ("new", f"new (<{MIN_CLAIM_WORDS} words, facts kept)")):
        s, u, c = totals[name]
        print(f"{label:<34}{s:>5} ({s / n:.1%}){u:>19}{c / n:>15.3f}")

    print(f"\nSentences newly counted as uncited claims: {len(newly_uncited)}")
    for question, sentence in newly_uncited[:args.show]:
        print(f"  - {sentence[:110]}")
        if question:
            print(f"      (question: {question[:80]})")


if __name__ == "__main__":
    main()