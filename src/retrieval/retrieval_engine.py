"""
The unified retrieval engine - this is what the rest of the
application calls. It orchestrates hybrid search + re-ranking
behin a clean interface.
"""

from src.retrieval.hybrid_retrieval import HybridRetriever, HybridResult
from src.retrieval.reranker import CrossEncoderReranker, RerankedResult
from src.ingestion.embedding_service import EmbeddingService

class RetrievalEngine:
    """Production retrieval: hybrid search -> re-rank -> return.
    
    Usage:
        engine = RetrievalEngine(vector_store, bm25, embedder)
        results = engine.retrieve("What's the refund policy?")

        for r in results:
            print(r.content, r.rerank_score)
    """

    def __init__(self, vector_store, bm25_retriever,
                 embedding_service, reranker=None,
                 dense_weight: float = 1.0,
                 sparse_weight: float = 1.0):
        
        self.embedder = embedding_service
        self.reranker = reranker

        self.hybrid = HybridRetriever(
            vector_store = vector_store,
            bm25_retriever = bm25_retriever,
            dense_weight = dense_weight,
            sparse_weight = sparse_weight,
        )

    def retrieve(self, query: str, top_k: int = 5,
                 use_reranker: bool = True,
                 metadata_filter: dict = None) -> list:
        """Full retrieval pipeline.
        1. Embed the query
        2. Hybrid search (dense + sparse + RRF)
        3. Optionally re-rank with cross-encoder
        
        
        the use_reranke flag lets you A/B test: does re-ranking
        actually improve your results? Measure it, don't assume.
        """

        # 1. Embed the query
        query_embedding = self.embedder.embed_text(query)

        # 2. Hybrid search - fetch more candidates than needed
        # so the re-ranker has good material to work with
        fetch_k = top_k * 4 if use_reranker else top_k

        hybrid_results = self.hybrid.search(
            query = query,
            query_embedding = query_embedding,
            top_k = fetch_k,
            metadata_filter = metadata_filter,
        )

        # 3. Re-rank if enabled and available
        if use_reranker and self.reranker and hybrid_results:
            return self.reranker.rerank(
                query = query,
                results = hybrid_results,
                top_k = top_k,
            )
        
        # Without re-ranking, return hybrid results directly
        return hybrid_results[:top_k]
    
    def retrieve_with_debug(self, query: str, 
                            top_k: int = 5) -> dict:
        """Retrieve with full debug info - invaluable during development and evaluation.
        
        Returns dense results, sparse results, fused results,
        and re-ranked results separately so you can see exactly
        what each stage contributes.
        """

        query_embedding = self.embedder.embed_text(query)

        # Run each stage independently for inspection
        dense = self.hybrid.vector_store.search(
            query_embedding, top_k = top_k
        )
        sparse = self.hybrid.bm25.search(query, top_k = top_k)
        hybrid = self.hybrid.search(
            query, query_embedding, top_k = top_k * 4
        )

        reranked = None
        if self.reranker:
            reranked = self.reranker.rerank(
                query, hybrid, top_k = top_k
            )

        return {
            "query": query,
            "dense_results": dense,
            "sparse_results": sparse,
            "hybrid_results": hybrid[:top_k],
            "reranked_results": reranked,
        }
    