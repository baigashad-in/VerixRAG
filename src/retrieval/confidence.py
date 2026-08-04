"""
Retrieval confidence gate: abstain BEFORE calling the LLM.

WHY:
If even the best retrieved chunk is a poor match, no prompt can
produce a grounded answer. Asking the LLM anyway costs a call and
invites it to improvise. The cross-encoder already scored every
chunk against the query, so we can decide deterministically:

    best rerank score < threshold  ->  "insufficient evidence"

The gate uses the RAW cross-encoder score (relevance), not any
domain-adjusted score, so source scoping can never trigger it.

CHOOSING THE THRESHOLD:
ms-marco-MiniLM logits are roughly -11 (unrelated) to +11 (exact
answer). The default (-6.0) only catches clearly unrelated queries.
Run `scripts/calibrate.py` to inspect the score distributions for 
in-scope and out-of-scope questions before increasing the threshold.
"""

from dataclasses import dataclass

ABSTAIN_ANSWER = ("I don't have enough information in the available "
                  "documents to answer this question.")


@dataclass
class GateDecision:
    passed: bool
    top_score: float | None
    threshold: float | None
    reason: str

    def as_dict(self) -> dict:
        return {"passed": self.passed, "top_score": self.top_score,
                "threshold": self.threshold, "reason": self.reason}


class ConfidenceGate:
    def __init__(self, min_score: float | None):
        """min_score=None disables the gate."""
        self.min_score = min_score

    def check(self, results: list) -> GateDecision:
        scores = [r.rerank_score for r in results
                  if getattr(r, "rerank_score", None) is not None]
        top = round(max(scores), 4) if scores else None

        if self.min_score is None:
            return GateDecision(True, top, None, "gate disabled")
        if top is None:
            # No cross-encoder scores (reranker off, or HyDE path): don't guess
            return GateDecision(True, None, self.min_score, "no rerank scores")
        if top < self.min_score:
            return GateDecision(False, top, self.min_score,
                                f"best rerank score {top} < {self.min_score}")
        return GateDecision(True, top, self.min_score, "above threshold")