"""
Builds the retrieval stack outside the API (scripts, benchmarks, calibration).

The API wires components in its lifespan; scripts need the same wiring
without starting a server.
Production components are measured in benchmarks.
"""

import os

from src.observability.tracing import measure, rss_mb


def build_retrieval_stack(with_reranker: bool = True) -> dict:
    """Load embedder, vector store, BM25 and (optionally) the reranker.

    Returns a dict with the components plus `startup_perf`, the load
    time and memory cost of each one.
    """
    from src.ingestion.embedding_service import EmbeddingService
    from src.storage.vector_store import VectorStore
    from src.retrieval.sparse_retrieval import BM25Retriever
    from src.retrieval.retrieval_engine import RetrievalEngine

    perf = {"rss_at_start_mb": rss_mb()}
    with measure("load_embedder", perf):
        embedder = EmbeddingService(
            model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    with measure("connect_vector_store", perf):
        store = VectorStore(connection_string=os.getenv("DATABASE_URL"),
                            embedding_dimension=embedder.dimension)
    bm25 = BM25Retriever()
    with measure("build_bm25_index", perf) as m:
        m["chunks"] = bm25.load_from_vector_store(store)

    reranker = None
    if with_reranker:
        from src.retrieval.reranker import CrossEncoderReranker
        with measure("load_reranker", perf):
            reranker = CrossEncoderReranker(model_name=os.getenv(
                "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))

    engine = RetrievalEngine(vector_store=store, bm25_retriever=bm25,
                             embedding_service=embedder, reranker=reranker)
    perf["rss_ready_mb"] = rss_mb()
    return {"embedder": embedder, "store": store, "bm25": bm25,
            "reranker": reranker, "engine": engine, "startup_perf": perf}