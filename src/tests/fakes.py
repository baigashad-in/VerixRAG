"""
Deterministic stand-ins for the ML models, so tests run offline.

- HashSentenceTransformer: bag-of-words hashing into 384 dims.
  Texts sharing words get similar vectors, which is enough to test
  dedup, domain centroids and retrieval plumbing.
- FakeCrossEncoder: logit from query/passage word overlap, on a
  scale similar to ms-marco-MiniLM (roughly -6 .. +4).
- fake_completion: a canned LiteLLM-shaped response with usage.

None of these say anything about the real models' quality — they
only make the plumbing testable without network or GPU.
"""

import hashlib
import re
from types import SimpleNamespace

import numpy as np

DIM = 384
_STOP = {
    "the", "a", "an", "is", "are", "to", "of", "and", "or", "in", "on", "for",
    "my", "i", "do", "does", "what", "how", "can", "you", "your", "with", "it",
    "be", "if", "at", "by", "from", "this", "that", "as", "will", "when",
}


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9][a-z0-9\-]*", text.lower()) if t not in _STOP]


class HashSentenceTransformer:
    def __init__(self, model_name: str = "fake", *args, **kwargs):
        self.model_name = model_name

    def get_embedding_dimension(self) -> int:
        return DIM

    get_sentence_embedding_dimension = get_embedding_dimension

    def _one(self, text: str, normalize: bool) -> np.ndarray:
        vec = np.zeros(DIM, dtype=np.float32)
        for tok in tokens(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % DIM] += 1.0 if (h >> 9) % 2 else -1.0
        norm = np.linalg.norm(vec)
        if normalize and norm > 0:
            vec /= norm
        return vec

    def encode(self, texts, normalize_embeddings: bool = False, **kwargs):
        if isinstance(texts, str):
            return self._one(texts, normalize_embeddings)
        return np.stack([self._one(t, normalize_embeddings) for t in texts]) \
            if texts else np.zeros((0, DIM), dtype=np.float32)


class FakeCrossEncoder:
    def __init__(self, model_name: str = "fake", *args, **kwargs):
        self.model_name = model_name

    def predict(self, pairs):
        scores = []
        for query, passage in pairs:
            q = set(tokens(query))
            p = set(tokens(passage))
            overlap = len(q & p) / len(q) if q else 0.0
            scores.append(10.0 * overlap - 6.0)
        return np.array(scores, dtype=np.float32)


class InMemoryVectorStore:
    """Minimal VectorStore stand-in: cosine search over a Python list."""

    def __init__(self):
        self.rows = []          # dicts: id, content, metadata, embedding
        self.version = 0

    def add(self, content: str, metadata: dict, embedding):
        self.rows.append({"id": len(self.rows) + 1, "content": content,
                          "metadata": dict(metadata),
                          "embedding": np.asarray(embedding, dtype=np.float32)})
        self.version += 1

    def search(self, query_embedding, top_k: int = 5, metadata_filter: dict = None):
        q = np.asarray(query_embedding, dtype=np.float32)
        rows = [r for r in self.rows
                if not metadata_filter
                or all(r["metadata"].get(k) == v for k, v in metadata_filter.items())]
        scored = sorted(rows, key=lambda r: float(np.dot(q, r["embedding"])), reverse=True)
        return [{"id": r["id"], "content": r["content"], "metadata": r["metadata"],
                 "similarity": float(np.dot(q, r["embedding"]))} for r in scored[:top_k]]

    def index_version(self) -> str:
        return f"mem-{self.version}"

    def domain_centroids(self) -> dict:
        groups = {}
        for r in self.rows:
            d = r["metadata"].get("domain")
            if d:
                groups.setdefault(d, []).append(r["embedding"])
        return {d: np.mean(v, axis=0).tolist() for d, v in groups.items()}

    def all_chunks(self):
        return [{"id": r["id"], "content": r["content"], "metadata": r["metadata"]}
                for r in self.rows]


def build_engine(docs: list[tuple[str, dict]], reranker=True):
    """RetrievalEngine over an in-memory store; docs = [(content, metadata)]."""
    from src.ingestion.embedding_service import EmbeddingService
    from src.retrieval.sparse_retrieval import BM25Retriever
    from src.retrieval.retrieval_engine import RetrievalEngine
    from src.retrieval.reranker import CrossEncoderReranker

    embedder = EmbeddingService.__new__(EmbeddingService)
    embedder.model_name = "fake"
    embedder.model = HashSentenceTransformer()
    embedder.dimension = DIM

    store = InMemoryVectorStore()
    for content, meta in docs:
        store.add(content, meta, embedder.embed_text(content))
    bm25 = BM25Retriever()
    bm25.index(store.all_chunks())

    rr = None
    if reranker:
        rr = CrossEncoderReranker.__new__(CrossEncoderReranker)
        rr.model = FakeCrossEncoder()
        rr.model_name = "fake"
    engine = RetrievalEngine(vector_store=store, bm25_retriever=bm25,
                             embedding_service=embedder, reranker=rr)
    return engine, store, embedder


def fake_completion(text: str, prompt_tokens: int = 1200, completion_tokens: int = 60):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens,
                              completion_tokens=completion_tokens,
                              total_tokens=prompt_tokens + completion_tokens),
    )
