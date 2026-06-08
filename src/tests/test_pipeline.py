"""
Integration tests for the full pipeline.

Requires:
- PostgreSQL with pgvector running (docker compose up db -d)
- Documents ingested (python scripts/ingest_docs.py)
- .env configured with DATABASE_URL and LLM API keys

Run with: pytest src/tests/test_pipeline.py -v -m integration -s

Skip with: pytest -m "not integration"
"""
import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

@pytest.fixture(scope="class")
def pipeline():
    """Full pipeline for all tests."""
    from src.ingestion.embedding_service import EmbeddingService
    from src.storage.vector_store import VectorStore
    from src.retrieval.sparse_retrieval import BM25Retriever
    from src.retrieval.reranker import CrossEncoderReranker
    from src.retrieval.retrieval_engine import RetrievalEngine
    from src.query.query_transformer import QueryTransformer
    from src.query.enhanced_retrieval import EnhancedRetriever
    from src.generation.generator import RAGGenerator
    from src.generation.hallucination_check import HallucinationDetector
    from src.pipeline import RAGPipeline
    from src.guardrails.guardrails_pipeline import GuardedRAGPipeline

    embedder = EmbeddingService()
    store = VectorStore(
        connection_string = os.getenv("DATABASE_URL"),
        embedding_dimension = embedder.dimension,
    )

    bm25 = BM25Retriever()
    bm25.load_from_vector_store(store)

    reranker = CrossEncoderReranker()
    engine = RetrievalEngine(store, bm25, embedder, reranker)

    transformer = QueryTransformer()
    enhanced = EnhancedRetriever(engine, transformer, embedder)

    generator = RAGGenerator()
    detector = HallucinationDetector()

    rag = RAGPipeline(enhanced, generator, detector)
    guarded = GuardedRAGPipeline(rag)

    yield guarded

    store.close()


@pytest.mark.integration
class TestRAGPipeline:
    
    def test_refund_query_returns_cited_answer(self, pipeline):
        """Basic query should return a cited answer from refund_policy.md."""
        result = pipeline.answer(query="What is the refund policy?")

        assert "answer" in result
        assert len(result["answer"]) > 20
        assert "refund" in result["answer"].lower()
        assert result["citations"] is not None
        assert len(result["citations"]) > 0

    def test_citations_reference_real_source(self, pipeline):
        """Citations should point to actual document files."""
        result = pipeline.answer(query = "How much does express shipping cost?")

        valid_sources = {"refund_policy.md", "shipping_policy.md", "account_guide.md", "policies.md"}
        for citation in result["citations"]:
            assert citation["source"] in valid_sources
            assert len(citation["preview"]) > 0

    def test_hallucination_check_passes(self, pipeline):
        """Faithfulness should be high for well-grounded answers."""
        result = pipeline.answer(query = "What is the refund policy?")

        h = result.get("hallucination_check")
        assert h is not None
        assert h["citation_coverage"] >= 0.5
        assert not h["is_suspicious"]

    def test_keyword_search_finds_error_code(self, pipeline):
        """BM25 should find exact keyword matches like error codes."""
        result = pipeline.answer(query = "error code E-4012")

        assert "E-4012" in result["answer"] or "e-4012" in result["answer"].lower()

    def test_multi_hop_query(self, pipeline):
        """Query requiring info from multiple documents."""
        result = pipeline.answer(query = "What happens to my store credit if I delete my account?")

        answer_lower = result["answer"].lower()
        assert "store credit" in answer_lower or "forfeited" in answer_lower
    
    def test_out_of_scope_rejected(self, pipeline):
        """Guardrails should reject off-topic queries."""
        result = pipeline.answer(query = "What is the weather in Tokyo?")

        assert "outside" in result["answer"].lower() or "can't help" in result["answer"].lower()
        assert result.get("guardrails", {}).get("classified_as") == "out_of_scope"

    def test_pii_detection(self, pipeline):
        """PII in query should be detected."""
        result = pipeline.answer(query = "My SSN is 123-45-6789, can I get a refund?")

        guardrails = result.get("guardrails", {})
        assert guardrails.get("input_pii_detected", 0) > 0

    def test_empty_result_handling(self, pipeline):
        """Query with no matching documents should fail gracefully."""
        result = pipeline.answer(query = "quantum entanglement in superconductors")

        assert "answer" in result
        # Should either return a "no info" message or a best-effort answer
        assert len(result["answer"]) > 0 
