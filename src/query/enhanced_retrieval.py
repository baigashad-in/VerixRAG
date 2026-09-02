"""
Retrieval with optional query transformation.

Each strategy only changes how CANDIDATES are found. Every strategy
then reranks, scopes and floors those candidates against the user's
ORIGINAL question, so the confidence gate and the score floor see
the same kind of score whatever the strategy. If the rewrite call
fails, retrieval falls back to the plain query.
"""

from enum import Enum

from src.observability.tracing import span


class QueryStrategy(Enum):
    """Which query transformation to apply."""
    NONE = "none"           # use query as-is
    MULTI_QUERY = "multi"   # break into sub-queries
    HYDE = "hyde"           # hypothetical document
    EXPAND = "expand"       # add related terms


class EnhancedRetriever:
    """Retrieval with query transformation support."""

    def __init__(self, retrieval_engine, query_transformer, embedding_service):
        self.engine = retrieval_engine
        self.transformer = query_transformer
        self.embedder = embedding_service

    def retrieve(self, query: str, top_k: int = 5,
                 strategy: QueryStrategy = QueryStrategy.NONE,
                 metadata_filter: dict = None) -> list:
        if strategy == QueryStrategy.NONE:
            return self.engine.retrieve(query, top_k=top_k, metadata_filter=metadata_filter)

        with span("retrieval.rewrite", strategy=strategy.value) as attrs:
            try:
                searches = self._searches(query, strategy)       # [(bm25 text, dense text)]
            except Exception as e:                               # fail open to the plain query
                attrs["fallback"] = f"{type(e).__name__}: {str(e)[:200]}"
                searches = None
            else:
                attrs["searches"] = len(searches)
        if searches is None:
            return self.engine.retrieve(query, top_k=top_k, metadata_filter=metadata_filter)

        fetch_k = self.engine._candidate_count(top_k, True)
        candidates, seen = [], set()
        for text, dense_text in searches:
            for r in self.engine.search_candidates(
                    text, self.embedder.embed_text(dense_text), fetch_k, metadata_filter):
                if r.chunk_id not in seen:
                    seen.add(r.chunk_id)
                    candidates.append(r)
        query_embedding = self.embedder.embed_text(query)
        return self.engine.rerank(query, candidates, top_k, query_embedding)

    def _searches(self, query: str, strategy: QueryStrategy) -> list[tuple[str, str]]:
        """(BM25 text, dense text) pairs to search for this strategy."""
        if strategy == QueryStrategy.HYDE:
            # BM25 keeps the original words; only the dense search uses the hypothetical answer
            return [(query, self.transformer.hyde(query))]
        if strategy == QueryStrategy.MULTI_QUERY:
            return [(q, q) for q in self.transformer.multi_query(query)]
        if strategy == QueryStrategy.EXPAND:
            expanded = self.transformer.expand_query(query)
            return [(expanded, expanded)]
        raise ValueError(f"Unknown strategy: {strategy}")