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
from sentence_transformers import CrossEncoder


@dataclass
class RerankedResult:
    """A result after cross-encoder re-scoring."""
    chunk_id: int
    content: str
    metadata: dict
    rerank_score: float             # cross-encoder relevance score
    original_rrf_score: float       # score before re-ranking
    final_score: float | None = None    # rerank_score after domain scoping
    domain_penalty: float = 0.0

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
                 model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 max_length: int | None = None):

        self.model = CrossEncoder(model_name)
        self.model_name = model_name
        if max_length is None:
            from src.config import get_settings
            max_length = get_settings().rerank_max_length
        if max_length:                                   # 0 keeps the model default
            self.max_length = max_length
        print(f"Loaded re-ranker: {model_name} (max length {self.max_length})")

    @property
    def max_length(self) -> int | None:
        """Tokens per query+passage pair; longer pairs are truncated."""
        model = getattr(self, "model", None)
        if model is None:
            return None
        if hasattr(type(model), "max_seq_length"):      # sentence-transformers >= 6
            return model.max_seq_length
        return getattr(model, "max_length", None)

    @max_length.setter
    def max_length(self, value: int):
        if value < 1:
            raise ValueError("max_length must be >= 1")
        if hasattr(type(self.model), "max_seq_length"):
            self.model.max_seq_length = value
        else:                                            # sentence-transformers < 6
            setattr(self.model, "max_length", value)

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