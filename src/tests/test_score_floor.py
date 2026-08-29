"""Per-chunk score floor: drop weak chunks before generation."""

from types import SimpleNamespace

import pytest

from src.config import reset_settings
from src.retrieval.confidence import apply_score_floor, chunk_score, floor_mask
from src.tests import fakes


def chunk(rerank, final=None):
    return SimpleNamespace(rerank_score=rerank, final_score=final)


def test_mask_keeps_best_even_below_floor():
    assert floor_mask([-9.0, -10.0], floor=-5.0, margin=None) == [True, False]


def test_margin_is_relative_to_best():
    assert floor_mask([6.0, 4.5, 1.0, 3.9], floor=None, margin=2.0) == [True, True, False, False]


def test_floor_and_margin_combine():
    assert floor_mask([2.0, 1.0, -3.0], floor=-2.0, margin=5.0) == [True, True, False]


def test_off_returns_input_unchanged():
    results = [chunk(1.0), chunk(-9.0)]
    assert apply_score_floor(results, None, None) is results
    assert apply_score_floor(results, None, 0) is results


def test_domain_adjusted_score_is_used():
    results = [chunk(5.0, final=5.0), chunk(4.0, final=2.0)]      # second one penalised
    assert chunk_score(results[1]) == 2.0
    assert apply_score_floor(results, None, 2.5) == results[:1]


def test_unscored_results_pass_through():
    results = [SimpleNamespace(rrf_score=0.1), SimpleNamespace(rrf_score=0.05)]
    assert apply_score_floor(results, 0.0, 1.0) == results


DOCS = [("Refunds take 14 days after the item arrives.", {"source": "a.md"}),
        ("Refunds for digital goods take 14 days.", {"source": "b.md"}),
        ("Express shipping costs fifteen dollars.", {"source": "c.md"}),
        ("Passwords must have eight characters.", {"source": "d.md"})]


def test_engine_applies_margin_from_settings(monkeypatch):
    monkeypatch.setenv("RERANK_CHUNK_MARGIN", "1")
    reset_settings()
    engine, _, _ = fakes.build_engine(DOCS)
    results = engine._retrieve_uncached("how long do refunds take", 4, True, None)
    scores = [r.rerank_score for r in results]
    assert len(results) < 4 and max(scores) - min(scores) <= 1
    assert engine.config_signature()[-1] == 1.0


def test_negative_margin_rejected(monkeypatch):
    monkeypatch.setenv("RERANK_CHUNK_MARGIN", "-1")
    reset_settings()
    from src.retrieval.retrieval_engine import RetrievalEngine
    with pytest.raises(ValueError):
        RetrievalEngine(vector_store=None, bm25_retriever=None, embedding_service=None)
