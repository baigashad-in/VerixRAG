"""
The unified retrieval engine - It orchestrates hybrid search + re-ranking.
"""

from src.retrieval.hybrid_retrieval import HybridRetriever
from src.cache import TTLCache, make_key, normalize_query
from src.observability.tracing import span, current_trace

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
                 sparse_weight: float = 1.0,
                 domain_scorer=None,
                 cache: TTLCache | None = None):
        
        self.embedder = embedding_service
        self.reranker = reranker
        self.domain_scorer = domain_scorer
        if cache is None:
            from src.config import get_settings
            s = get_settings()
            cache = (TTLCache(s.cache_max_entries, s.cache_ttl_seconds, name="retrieval")
                     if s.cache_enabled else None)
        self.cache = cache

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
        actually improve your results?
        """

        # 0. Retrieval cache: same normalised query against the same index
        cache_key = None
        if self.cache is not None:
            cache_key = self._cache_key(query, top_k, use_reranker, metadata_filter)
            with span("retrieval.cache_lookup") as attrs:
                hit, cached, age = self.cache.get(cache_key)
                attrs["hit"] = hit
            trace = current_trace()
            if hit:
                if trace is not None:
                    trace.set(retrieval_cache="hit", retrieval_cache_age_s=age,
                              scoping=cached["scoping"])
                return cached["results"]

        results = self._retrieve_uncached(query, top_k, use_reranker, metadata_filter)
        if cache_key is not None:
            trace = current_trace()
            self.cache.set(cache_key, {
                "results": results,
                "scoping": trace.attrs.get("scoping") if trace is not None else None,
            })
        return results

    def _cache_key(self, query, top_k, use_reranker, metadata_filter) -> str:
        store = self.hybrid.vector_store
        version = store.index_version() if hasattr(store, "index_version") else "na"
        scoping = ((self.domain_scorer.enabled, self.domain_scorer.penalty,
                    self.domain_scorer.min_margin) if self.domain_scorer else None)
        return make_key("retrieval", normalize_query(query), top_k, use_reranker,
                        metadata_filter, version, scoping)

    def _retrieve_uncached(self, query: str, top_k: int,
                           use_reranker: bool, metadata_filter: dict) -> list:
        # 1. Embed the query
        with span("retrieval.embed_query"):
            query_embedding = self.embedder.embed_text(query)

        # 2. Hybrid search - fetch more candidates than needed
        # so the re-ranker has good material to work with
        fetch_k = top_k * 4 if use_reranker else top_k

        with span("retrieval.hybrid_search", fetch_k=fetch_k) as attrs:
            hybrid_results = self.hybrid.search(
                query = query,
                query_embedding = query_embedding,
                top_k = fetch_k,
                metadata_filter = metadata_filter,
            )
            attrs["candidates"] = len(hybrid_results)

        # 3. Re-rank if enabled and available
        if use_reranker and self.reranker and hybrid_results:
            # Score ALL candidates, so domain scoping can promote an
            # in-domain chunk from below the cut-off.
            with span("retrieval.rerank", candidates=len(hybrid_results)) as attrs:
                reranked = self.reranker.rerank(
                    query = query,
                    results = hybrid_results,
                    top_k = len(hybrid_results),
                )
                attrs["top_score"] = (round(reranked[0].rerank_score, 3)
                                      if reranked else None)

            # 4. Source scoping: down-rank chunks from other domains
            if self.domain_scorer is not None:
                with span("retrieval.scoping") as attrs:
                    detection = self.domain_scorer.detect(query, query_embedding)
                    reranked = self.domain_scorer.apply(reranked, detection)
                    attrs.update({k: v for k, v in detection.items() if k != "scores"})
                    attrs["penalized_in_top_k"] = sum(
                        1 for r in reranked[:top_k] if r.domain_penalty)
                trace = current_trace()
                if trace is not None:
                    trace.set(scoping=detection)
            return reranked[:top_k]
        
        # Without re-ranking, return hybrid results directly
        return hybrid_results[:top_k]

    
    def refresh_domains(self):
        """Reload per-domain centroids from the store (startup / after ingest)."""
        if self.domain_scorer is not None and hasattr(self.hybrid.vector_store, "domain_centroids"):
            self.domain_scorer.refresh(self.hybrid.vector_store.domain_centroids())

    def clear_cache(self):
        if self.cache is not None:
            self.cache.clear()

    def cache_stats(self) -> dict | None:
        return self.cache.stats() if self.cache is not None else None

    def on_index_changed(self):
        """Called by the API after /api/ingest."""
        store = self.hybrid.vector_store
        if hasattr(store, "invalidate_version"):
            store.invalidate_version()
        self.clear_cache()
        self.refresh_domains()
        
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