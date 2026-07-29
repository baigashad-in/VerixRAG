"""
Evaluation runner - measures the RAG pipeline end-to-end.

Every score in the report is traceable: each example stores the
generated answer, the chunks it was judged against, and the raw
judge replies. Judge failures and refusals are counted separately
instead of being folded into the averages as 0.0.
"""

from dataclasses import dataclass, field
from typing import Optional
import json
import time
from pathlib import Path
from datetime import datetime

from .metrics import JudgeError

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")

import re
REFUSAL = re.compile(r"(don't|do not) have enough information|could(n't| not) find any relevant", re.I)


@dataclass
class EvalResult:
    question: str
    category: str
    generated_answer: str
    ground_truth: str
    retrieved_contexts: list[str]
    contexts_source: str
    outcome: str = "answered"   # answered | refused | empty
    scores: dict = field(default_factory=dict)        # metric -> float or None
    judge_errors: dict = field(default_factory=dict)  # metric -> error text
    judge_raw: dict = field(default_factory=dict)     # metric -> raw judge replies
    claims: Optional[tuple] = None                    # (supported, total)


class EvaluationRunner:
    def __init__(self, rag_pipeline, metrics, eval_dataset):
        self.pipeline = rag_pipeline
        self.metrics = metrics
        self.dataset = eval_dataset

    def _contexts_from_output(self, output: dict):
        """Use the chunks the generator actually saw, if the pipeline returns them."""
        for key in ("retrieval_results", "sources", "contexts", "chunks"):
            val = output.get(key)
            if isinstance(val, list) and val:
                first = val[0]
                if hasattr(first, "content"):
                    return [r.content for r in val], f"pipeline:{key}"
                if isinstance(first, dict) and "content" in first:
                    return [r["content"] for r in val], f"pipeline:{key}"
        return None, None

    def _score(self, fn, *args):
        self.metrics.trace = []
        self.metrics.last_claims = None
        try:
            value, error = fn(*args), None
        except Exception as e:  # judge failure, rate limit, anything
            value, error = None, f"{type(e).__name__}: {e}"[:300]
        return value, error, list(self.metrics.trace)

    def run(self, config_label: str = "default") -> dict:
        results = []
        start = time.time()
        n = len(self.dataset.examples)
        print(f"\nRunning evaluation: {config_label} ({n} examples)", flush=True)

        for i, ex in enumerate(self.dataset.examples):
            print(f"\n[{i+1}/{n}] {ex.question[:60]}", flush=True)
            output = self.pipeline.answer(query=ex.question, check_hallucination=False, return_chunks=True)
            answer = output.get("answer", "")

            contexts, source = self._contexts_from_output(output)
            if contexts is None:
                retrieved = self.pipeline.retriever.retrieve(query=ex.question, top_k=8)
                contexts, source = [r.content for r in retrieved], "re-retrieved"

            r = EvalResult(
                question=ex.question,
                category=(ex.metadata or {}).get("category", "unknown"),
                generated_answer=answer,
                ground_truth=ex.ground_truth_answer,
                retrieved_contexts=contexts,
                contexts_source=source,
            )

            if not answer.strip():
                r.outcome = "empty"
            elif REFUSAL.search(answer):
                r.outcome = "refused"

            jobs = {
                "faithfulness": (self.metrics.faithfulness, answer, contexts),
                "answer_relevancy": (self.metrics.answer_relevancy, ex.question, answer),
                "context_precision": (self.metrics.context_precision, ex.question,
                                      contexts, ex.ground_truth_answer),
                "context_recall": (self.metrics.context_recall,
                                   ex.ground_truth_answer, contexts),
            }
            for name, (fn, *args) in jobs.items():
                if r.outcome in ("empty", "refused") and name in ("faithfulness", "answer_relevancy"):
                    r.scores[name] = None
                    print(f"  {name}: skipped ({r.outcome})", flush=True)
                    continue
                value, error, raw = self._score(fn, *args)
                r.scores[name] = value
                r.judge_raw[name] = raw
                if error:
                    r.judge_errors[name] = error
                if name == "faithfulness":
                    r.claims = getattr(self.metrics, "last_claims", None)
                shown = "FAILED" if error else ("n/a" if value is None else f"{value:.3f}")
                print(f"  {name}: {shown}", flush=True)

            results.append(r)

        report = self._build_report(results, config_label, time.time() - start)
        self._print_report(report)
        return report

    def _build_report(self, results, config_label, elapsed) -> dict:
        def mean(vals):
            vals = [v for v in vals if v is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        in_scope = [r for r in results if r.category not in ("out_of_scope",)]
        out_scope = [r for r in results if r.category == "out_of_scope"]

        supported = sum(r.claims[0] for r in in_scope if r.claims)
        total = sum(r.claims[1] for r in in_scope if r.claims)
        refused = [r for r in in_scope if r.outcome == "refused"]
        empty = [r for r in in_scope if r.outcome == "empty"]

        in_scope_means = {m: mean(r.scores.get(m) for r in in_scope) for m in METRICS}

        return {
            "config": config_label,
            "timestamp": datetime.now().isoformat(),
            "num_examples": len(results),
            "elapsed_seconds": round(elapsed, 1),
            "in_scope": {
                "n": len(in_scope),
                "hallucination_rate": round(1 - supported / total, 4) if total else None,
                "claims_supported": supported,
                "claims_total": total,
                "answered": sum(1 for r in in_scope if r.outcome == "answered"),
                "refused": len(refused),
                "empty": len(empty),
                **in_scope_means,
                "scored_counts": {m: sum(1 for r in in_scope if r.scores.get(m) is not None)
                                  for m in METRICS},
            },
            "out_of_scope": {
                "n": len(out_scope),
                "refused": sum(1 for r in out_scope if r.outcome == "refused"),
            },
            "judge_failures": sum(len(r.judge_errors) for r in results),
            "contexts_re_retrieved": sum(1 for r in results if r.contexts_source == "re-retrieved"),
            "aggregate": in_scope_means,  # kept for compatibility
            "per_example": [
                {
                    "question": r.question,
                    "category": r.category,
                    "outcome": r.outcome,
                    "answer": r.generated_answer,
                    "contexts_source": r.contexts_source,
                    "contexts": [c[:600] for c in r.retrieved_contexts],
                    "scores": r.scores,
                    "claims": r.claims,
                    "judge_errors": r.judge_errors,
                    "judge_raw": r.judge_raw,
                }
                for r in results
            ],
        }

    def _print_report(self, report: dict):
        s = report["in_scope"]
        print(f"\n{'='*50}\nEVALUATION REPORT: {report['config']}\n{'='*50}")
        print(f"Examples: {report['num_examples']}   Time: {report['elapsed_seconds']}s")
        print(f"In-scope: {s['n']}  answered: {s['answered']}  refused: {s['refused']}  empty: {s['empty']}")
        print(f"Hallucination rate: {s['hallucination_rate']}  "
              f"({s['claims_supported']}/{s['claims_total']} claims supported)")
        for m in METRICS:
            print(f"{m:<20} {s[m]}  (scored {s['scored_counts'][m]}/{s['n']})")
        print(f"Out-of-scope refused: {report['out_of_scope']['refused']}/{report['out_of_scope']['n']}")
        print(f"Judge failures: {report['judge_failures']}   "
              f"Re-retrieved contexts: {report['contexts_re_retrieved']}")
        print("=" * 50, flush=True)

    def save_report(self, report: dict, directory: str = "./experiments"):
        Path(directory).mkdir(parents=True, exist_ok=True)
        stamp = report["timestamp"][:19].replace(":", "-")
        filepath = Path(directory) / f"{report['config']}_{stamp}.json"
        filepath.write_text(json.dumps(report, indent=2))
        print(f"Report saved: {filepath}")
