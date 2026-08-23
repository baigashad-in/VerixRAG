"""Standard IR metrics against graded relevance labels (BEIR/TREC qrels)."""

from __future__ import annotations

import math

Qrels = dict[str, int]          # doc_id -> relevance grade (0 = not relevant)


def ndcg_at_k(ranked: list[str], qrels: Qrels, k: int) -> float:
    """NDCG@k with linear gain, as computed by trec_eval / pytrec_eval."""
    dcg = sum(qrels.get(doc, 0) / math.log2(i + 2) for i, doc in enumerate(ranked[:k]))
    ideal = sorted((g for g in qrels.values() if g > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: list[str], qrels: Qrels, k: int) -> float:
    relevant = {doc for doc, g in qrels.items() if g > 0}
    if not relevant:
        return 0.0
    return len(relevant & set(ranked[:k])) / len(relevant)


def mrr_at_k(ranked: list[str], qrels: Qrels, k: int) -> float:
    for i, doc in enumerate(ranked[:k]):
        if qrels.get(doc, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def overlap_at_k(approximate: list[str], exact: list[str], k: int) -> float:
    """Share of the exact top-k that the approximate search also returned."""
    truth = set(exact[:k])
    return len(truth & set(approximate[:k])) / len(truth) if truth else 1.0


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None
