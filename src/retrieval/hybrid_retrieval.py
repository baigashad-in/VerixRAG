"""
Hybrid retrieval: merge dense + sparse results with RPF.

THE MERGING PROBLEM:
- Vector search returns similarity scores between 0 and 1
- BM25 returns unbounded relevance scores (could be 0.5 or 50)
- You can't just add them - the scales are completely different

RRF SOLVES THIS by ignoring scores entirely and using only RANKS:

    RRF_score(doc) = Σ 1 / (k + rank_in_list)

    where k is a constant (typically 60) that controls how much
    we penalize lower-ranked items.

WHY k=60?
Research paper (Cormack et al., 2009) found 60 works well across
many datasets. Lower k = top results dominate. Higher k = more
equal weighting across ranks. 60 is a well-tested default.

EXAMPLE:
    Doc A: rank 1 in vector, rank 3 in BM25
    RRF = 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323

    Doc B: rank 2 in vector, not in BM25
    RRF = 1/(60+2) + 0 = 0.0161

    Doc C: not in vector, rank 1 in BM25
    RRF = 0 + 1/(60+1) = 0.0164

    Final ranking: A > C > B
    Doc A wins because it appears in BOTH lists.
"""

from dataclasses import dataclass
from src.retrieval.sparse_retrieval import BM25Retriever, SparseResult
from src.storage.vector_store import VectorStore

@dataclass
class HybridResult:
    """A merged search result with its fusion score."""
    chunk_id: int
    content: str
    metadata: dict
    rrf_score: float
    dense_rank: int | None   # rank in vector search (None if absent)
    sparse_rank: int | None  # rank in BM25 search (None if absent)

class HybridRetriever:
    """Combines dense vector search and sparse BM25 search.
    
    This is the retrieval engine your RAG pipeline actually uses.
    Neither dense nor sparse search alone is optimal - hybrid
    consistently outperforms both in benchmarks.
    """

    def __init__(self, vector_store, bm25_retriever,
                 rrf_k: int = 60,
                 dense_weight: float = 1.0,
                 sparse_weight: float = 1.0):
        
        """
        Args:
            rrf_k: RRF constant. 60 is standard.
            dense_weight: multiplier for vector search RRF scores.
                        Set > 1 to favor semantic search.
            sparse_weight: multiplier for BM25 RRF scores.
                        Set > 1 to favor keyword search.

        Weights let you tune the balance. For technical docs with
        lots of codes/IDs, you might boost sparse_weight. For
        conversational queries, boost dense_weight.
        """

        self.vector_store = vector_store
        self.bm25 = bm25_retriever
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def search(self, query: str, query_embedding: list[float],
               top_k: int = 5,
               fetch_k: int = 20,
               metadata_filter: dict = None) -> list[HybridResult]:
        
        """Run hybrid search: dense + sparse, merged with RRF.
        
        Args:
            query: raw text query (for BM25)
            query_embedding: embedded query vector (for dense search)
            top_k: number of final results to return
            fetch_k: how many results to fetch from EACH source
                     before merging. Always fetch more than top_k
                     so RRF has enough candidates to work with.
            metadata_filter: optional filter passed to vector search
            """
        
        # 1. Get results from both retrieval methods
        dense_results = self.vector_store.search(
            query_embedding, 
            top_k = fetch_k,
            metadata_filter = metadata_filter,
        )

        sparse_results = self.bm25.search(query, top_k = fetch_k)

        # 2. Build RRF scores
        # Key: chunk_id -> {content, metadata, rrf_score, ranks}
        fused = {}

        # Score dense results by their rank
        for rank, result in enumerate(dense_results):
            chunk_id = result["id"]
            rrf_contribution = (
                self.dense_weight * (1.0 / (self.rrf_k + rank + 1))
            )

            if chunk_id not in fused:
                fused[chunk_id] = {
                    "content": result["content"],
                    "metadata": result["metadata"],
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "sparse_rank": None,
                }
            
            fused[chunk_id]["rrf_score"] += rrf_contribution
            fused[chunk_id]["dense_rank"] = rank + 1

        # Score sparse results by their rank
        for rank, result in enumerate(sparse_results):
            chunk_id = result.chunk_id
            rrf_contribution = (
                self.sparse_weight * (1.0 / (self.rrf_k + rank + 1))
            )

            if chunk_id not in fused:
                fused[chunk_id] = {
                    "content": result.content,
                    "metadata": result.metadata,
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "sparse_rank": None,
                }

            fused[chunk_id]["rrf_score"] += rrf_contribution
            fused[chunk_id]["sparse_rank"] = rank + 1

        # 3. Sort by fused score and return top_k
        ranked = sorted(
            fused.items(),
            key = lambda x: x[1]["rrf_score"],
            reverse = True
        )[:top_k]

        return [
            HybridResult(
                chunk_id = chunk_id,
                content = data["content"],
                metadata = data["metadata"],
                rrf_score = data["rrf_score"],
                dense_rank = data["dense_rank"],
                sparse_rank = data["sparse_rank"],
            )
            for chunk_id, data in ranked
        ]