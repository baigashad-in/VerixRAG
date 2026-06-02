"""Tests for retrieval components."""

import pytest
from src.retrieval.sparse_retrieval import BM25Retriever

class TestBM25Retriever:

    @pytest.fixture
    def indexed_retriever(self):
        retriever = BM25Retriever()
        chunks = [
            {"id": 1, "content": "Refund policy allows returns within 30 days", "metadata": {}},
            {"id": 2, "content": "Express shipping costs fifteen dollars", "metadata": {}},
            {"id": 3, "content": "Account deletion removes all personal data", "metadata": {}},
            {"id": 4, "content":"Store credit never expires after refund", "metadata": {}},
            {"id": 5, "content": "Password must contain eight characters minimum", "metadata": {}},
            {"id": 6, "content": "Two factor authentication enhances security", "metadata": {}},
            {"id": 7, "content": "Order tracking is available for all shipments", "metadata": {}},
        ]
        retriever.index(chunks)
        return retriever
    
    def test_index_builds(self, indexed_retriever):
        assert indexed_retriever.bm25 is not None
        assert len(indexed_retriever.documents) == 7

    def test_search_returns_results(self, indexed_retriever):
        results = indexed_retriever.search("refund policy", top_k=2)
        assert len(results) > 0
        assert len(results) <= 2

    def test_relevant_result_first(self, indexed_retriever):
        results = indexed_retriever.search("refund", top_k=3)
        # Chunks about refund should score higher
        contents = [r.content for r in results]
        assert any("refund" in c.lower() for c in contents)

    def test_search_before_index_raises(self):
        retriever = BM25Retriever()
        with pytest.raises(RuntimeError):
            retriever.search("test query")

    def test_no_results_for_unrelated_query(self, indexed_retriever):
        results = indexed_retriever.search("quantum physics", top_k=3)
        # Should return empty or very low scores
        for r in results:
            assert r.score < 1.0