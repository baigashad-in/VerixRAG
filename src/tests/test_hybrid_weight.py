"""Hybrid BM25 weight: setting, validation and cache isolation."""

import pytest

from src.config import reset_settings
from src.tests import fakes

DOCS = [
    ("Refunds are processed within 14 days of the return.", {"source": "a.md"}),
    ("Error E-4012 means the payment card was declined.", {"source": "b.md"}),
]


def test_default_weight_comes_from_settings(monkeypatch):
    monkeypatch.setenv("HYBRID_SPARSE_WEIGHT", "0.3")
    reset_settings()
    engine, _, _ = fakes.build_engine(DOCS, reranker=False)
    assert engine.hybrid.sparse_weight == 0.3
    assert engine.hybrid.dense_weight == 1.0


def test_explicit_weight_overrides_setting(monkeypatch):
    from src.retrieval.retrieval_engine import RetrievalEngine
    monkeypatch.setenv("HYBRID_SPARSE_WEIGHT", "0.3")
    reset_settings()
    engine = RetrievalEngine(vector_store=None, bm25_retriever=None,
                             embedding_service=None, sparse_weight=0.7)
    assert engine.hybrid.sparse_weight == 0.7


def test_negative_weight_rejected():
    from src.retrieval.retrieval_engine import RetrievalEngine
    with pytest.raises(ValueError):
        RetrievalEngine(vector_store=None, bm25_retriever=None,
                        embedding_service=None, sparse_weight=-0.1)


def test_cache_key_changes_with_weight():
    engine, _, _ = fakes.build_engine(DOCS, reranker=False)
    before = engine._cache_key("refund", 5, False, None)
    engine.hybrid.sparse_weight = 0.3
    assert engine._cache_key("refund", 5, False, None) != before


def test_weight_changes_fusion_order():
    engine, _, embedder = fakes.build_engine(DOCS, reranker=False)
    query = "E-4012 refunds"
    emb = embedder.embed_text(query)
    engine.hybrid.sparse_weight = 0.0
    dense_only = [r.chunk_id for r in engine.hybrid.search(query, emb, top_k=2)]
    dense = [r["id"] for r in engine.hybrid.vector_store.search(emb, top_k=2)]
    assert dense_only == dense                          # weight 0 = pure dense ranking