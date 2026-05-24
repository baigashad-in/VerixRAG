"""
Cross-encoder re-ranking for improved rerieval precision.

WHY RE-RANK?
Embedding models (bi-encoders) encode query and document SEPARATELY,
then compare. This is fast but loses nuance.

Cross-encoders process query AND document TOGETHER, capturing
fine-grained interactions between them. Much more accurate,
but too slow to run against your entire database.

THE PATTERN:
1. Hybrid search fetches 20 candidates (fast, broad)
2. Cross-encoder re-scores those 20 (slow, precise)
3. Return top 5 after re-ranking

This two-stage pipeline (retrieve broadly -> re-rank precisely) 
is the standard pattern at Google, Bing, and every serious
search system.
"""

from dataclasses import dataclass
from src.retrieval.hybrid_retrieval import HybridResult

@dataclass
class RerankedResult:
    """A result after cross-encoder re-scoring."""
    chunk_id: int
    content: str
    metadata: dict
    rerank_score: float             # cross-encoder relevance score
    original_rrf_score: float       # score before re-ranking

class CrossEncoderReranker:
    """Re-ranks retrieved results using a cross-encoder model.
    
    MODEL CHOICES:
    - cross-encoder/ms-marco-MiniLM-L-6-v2: fast, good quality
    - BAAI/bge-reranker-base: strong open-source option
    - Cohere Rerank API: best quality, costs money

    We use ms-marco-MiniLM because it's small enough to run
    locally but trained specifically for passage re-ranking.
    """

    def __init__(self,
                 model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)
        self.model_name = model_name
        print(f"Loaded re-ranker: {model_name}")

    def rerank(self, query: str,
               results: list[HybridResult],
               top_k: int = 5) -> list[RerankedResult]:
        """Re-score results based on query-document interaction.

        The cross-encoder sees BOTH the query and document text
        together, so it can understand relationships like:

        Query: "Can I return after 30 days?"
        Doc: "After 30 days, store credit is offered."
        -> High score: the cross-encoder understands this ANSWERS
        the question, even though it doesn't contain "return"

        A bi-encoder might rank this lower because the word
        "return" is missing from the document.
        """
        if not results:
            return []
        
        # Create (query, document) pairs for the cross-encoder
        pairs = [
            (query, result.content)
            for result in results
        ]

        # Score all pairs at once
        scores = self.model.predict(pairs)

        # Attach scores and re-sort
        reranked = []
        for result, score in zip(results, scores):
            reranked.append(RerankedResult(
                chunk_id = result.chunk_id,
                content = result.content,
                metadata = result.metadata,
                rerank_score = float(score),
                original_rrf_score = result.rrf_score,
            ))

        # Sort by cross-encoder score (highest = most relevant)
        reranked.sort(key=lambda r: r.rerank_score, reverse=True)

        return reranked[:top_k]