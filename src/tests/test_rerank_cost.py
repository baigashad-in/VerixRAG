"""Reranker cost settings: candidate count and token limit."""

import pytest

from src.config import reset_settings
from src.tests import fakes

DOCS = [(f"Refund rule {i}: refunds take {i} days for order type {i}.", {"source": f"d{i}.md"})
        for i in range(12)]


def scored(engine, query, top_k):
    """Run the engine and return how many candidates the reranker scored."""
    seen = []
    original = engine.reranker.rerank

    def spy(query, results, top_k):
        seen.append(len(results))
        return original(query, results, top_k)

    engine.reranker.rerank = spy
    engine._retrieve_uncached(query, top_k, True, None)
    return seen[0]


def test_default_scores_four_times_top_k():
    engine, _, _ = fakes.build_engine(DOCS)
    assert scored(engine, "refund days", 2) == 8


def test_candidate_setting_limits_reranker_input(monkeypatch):
    monkeypatch.setenv("RERANK_CANDIDATES", "5")
    reset_settings()
    engine, _, _ = fakes.build_engine(DOCS)
    assert engine.rerank_candidates == 5
    assert scored(engine, "refund days", 2) == 5
    assert scored(engine, "refund days", 7) == 7            # never fewer than top_k


def test_negative_candidates_rejected():
    from src.retrieval.retrieval_engine import RetrievalEngine
    with pytest.raises(ValueError):
        RetrievalEngine(vector_store=None, bm25_retriever=None, embedding_service=None,
                        rerank_candidates=-1)


def test_no_reranker_fetches_only_top_k():
    engine, _, _ = fakes.build_engine(DOCS, reranker=False)
    assert engine._candidate_count(3, use_reranker=True) == 3


def test_max_length_truncates_and_changes_cache_key():
    engine, _, _ = fakes.build_engine(DOCS)
    before = engine.config_signature()
    engine.reranker.max_length = 3
    assert engine.reranker.model.max_length == 3
    assert engine.config_signature() != before
    fake = fakes.FakeCrossEncoder()
    fake.max_length = 3
    passage = "x " * 10 + "refund"                  # the matching word sits past the limit
    assert fake.predict([("refund", passage)])[0] == pytest.approx(-6.0)


def test_max_length_rejects_zero():
    engine, _, _ = fakes.build_engine(DOCS)
    with pytest.raises(ValueError):
        engine.reranker.max_length = 0


def test_max_length_setting_applied_on_load(monkeypatch):
    import sys
    from src.retrieval.reranker import CrossEncoderReranker
    monkeypatch.setattr(sys.modules["sentence_transformers"], "CrossEncoder",
                        fakes.FakeCrossEncoder, raising=False)
    monkeypatch.setenv("RERANK_MAX_LENGTH", "128")
    reset_settings()
    assert CrossEncoderReranker(model_name="fake").max_length == 128
    monkeypatch.setenv("RERANK_MAX_LENGTH", "0")
    reset_settings()
    assert CrossEncoderReranker(model_name="fake").max_length == 512      # model default