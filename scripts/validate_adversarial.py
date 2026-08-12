"""
Check the adversarial dataset against the documents.

Every fact in must_include_any must actually appear in the knowledge
base, otherwise the case would fail for the wrong reason. Run this
after editing documents/ or datasets/adversarial_eval.json.

Usage:
    python -m scripts.validate_adversarial
    python -m scripts.validate_adversarial --documents ./documents --dataset datasets/adversarial_eval.json
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.adversarial import load_cases, normalize_answer


def validate(documents_dir: str, dataset: str) -> list[str]:
    corpus = "\n".join(
        normalize_answer(p.read_text(encoding="utf-8")).lower()
        for p in sorted(Path(documents_dir).rglob("*"))
        if p.suffix in (".md", ".txt") and not p.name.startswith(".")
    )
    if not corpus:
        return [f"no documents found in {documents_dir}"]
    problems = []
    for case in load_cases(dataset):
        if case.must_include_any and not any(s.lower() in corpus for s in case.must_include_any):
            problems.append(f"{case.id}: none of {case.must_include_any} appear in the documents")
    return problems


def main():
    parser = argparse.ArgumentParser(description="Validate adversarial cases against documents")
    parser.add_argument("--documents", default="./documents")
    parser.add_argument("--dataset", default="datasets/adversarial_eval.json")
    args = parser.parse_args()

    problems = validate(args.documents, args.dataset)
    if problems:
        print("Problems found:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print(f"OK: every expected fact in {args.dataset} appears in {args.documents}")


if __name__ == "__main__":
    main()