"""
Retrieval engine enhanced with query transformation.

This wraps the hybrid retriever from Phase 3 and adds
query preprocessing. The key architectural decision:
query transformation is OPTIONAL and CONFIGURABLE,
because each strategy adds latency (an LLM call) and
may not help for simple queries.
"""

from enum import Enum

class QueryStrategy(Enum):
    """Which query transformation to apply."""
    NONE = "none"           # use query as-is
    MULTI_QUERY = "multi"   # break into sub-queries
    HYDE = "hyde"           # hypothetical document
    EXPAND = "expand"       # add related terms


class EnhancedRetriever:
    """Retrieval with query transformation support.
    
    ARCHITECTURE DECISION:
    Simple queries ("What is X?") don't need transformation.
    Complex queries ("Compare X, Y, and Z") benefit from multi-query.
    Vague queries ("the thing that does stuff") benefit from HyDE.
    """

    def __init__(self, retrieval_engine, query_transformer,
                 embedding_service):
        self.engine = retrieval_engine
        self.transformer = query_transformer
        self.embedder = embedding_service

    def retrieve(self, query: str, top_k: int = 5,
                 strategy: QueryStrategy = QueryStrategy.NONE,
                 metadata_filter: dict = None) -> list:
        
        if strategy == QueryStrategy.NONE:
            return self.engine.retrieve(
                query, top_k=top_k,
                metadata_filter = metadata_filter, 
            )
        
        elif strategy == QueryStrategy.HYDE:
            return self._retrieve_with_hyde(
                query, top_k, metadata_filter
            )
        
        elif strategy == QueryStrategy.MULTI_QUERY:
            return self._retrieve_multi_query(
                query, top_k, metadata_filter
            )
        
        elif strategy == QueryStrategy.EXPAND:
            expanded = self.transformer.expand_query(query)
            return self.engine.retrieve(
                expanded, top_k=top_k,
                metadata_filter = metadata_filter,
            )
        
    def _retrieve_with_hyde(self, query, top_k, metadata_filter):
        """Generate hypothetical answer, embed it, search with that.
        
        We still pass the ORIGINAL query to BM25 (keyword search),
        because the hypothetical answer might not contain the exact
        keywords. HyDE only transforms the dense search embedding.
        """

        hypothetical = self.transformer.hyde(query)
        hyde_embedding = self.embedder.embed_text(hypothetical)

        # Use hypothetical embedding for dense, original for sparse
        return self.engine.hybrid.search(
            query = query,                      # original for BM25
            query_embedding = hyde_embedding,   # hyde for vectors
            top_k = top_k,
            metadata_filter = metadata_filter,
        )
    
    def _retrieve_multi_query(self, query, top_k, metadata_filter):
        """Retrieve for each sub-query, merge and deduplicate.
        
        Each sub-query gets its own retrieval pass. We merge all
        results and deduplicate by chunk_id, keeping the highest
        score for each unique chunk.
        """

        sub_queries = self.transformer.multi_query(query)

        all_results = {}    # chunk_id -> best result

        for sub_q in sub_queries:
            results = self.engine.retrieve(
                sub_q, top_k=top_k,
                metadata_filter = metadata_filter,
            )

            for r in results:
                cid = r.chunk_id
                # Keep the result with the highest score
                if cid not in all_results or \
                    r.rerank_score > all_results[cid].rerank_score:
                        all_results[cid] = r

        # Sort merged results by score, return top_k
        merged = sorted(
            all_results.values(),
            key = lambda r: r.rerank_score,
            reverse = True,
        )

        return merged[:top_k]
