"""BM25 keyword retrieval over a sparse inverted index (numpy, no per-document loops at query time)."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

try:
    import Stemmer
except ImportError:                    # stemming is optional; scores are slightly lower without it
    Stemmer = None

# Lucene / Elasticsearch English stopwords (the BEIR BM25 baseline uses the same list).
STOPWORDS = frozenset("""
a an and are as at be but by for if in into is it no not of on or such that the their
then there these they this to was will with
""".split())
_DROP = STOPWORDS | frozenset("bcdefghijklmnopqrstuvwxyz")     # plus single letters

_WORD = re.compile(r"[^\W_]+")
_COMPOUND = re.compile(r"[^\W_]+(?:[-_][^\W_]+)+")


@dataclass
class SparseResult:
    """A BM25 search result."""
    chunk_id: int
    content: str
    metadata: dict
    score: float


@dataclass
class _Index:
    """Immutable index snapshot; swapped in one assignment so searches never see a partial build."""
    vocab: dict[str, int]
    indptr: np.ndarray          # per term: slice into doc_idx / weights
    doc_idx: np.ndarray         # int32 document positions
    weights: np.ndarray         # float32 precomputed BM25 term weights
    chunk_ids: np.ndarray       # position -> chunk id
    num_docs: int
    documents: list | None      # in-memory chunks, or None to fetch hits from the store
    store: object | None


class BM25Retriever:
    """BM25 (Lucene variant) with stemming, stopwords and code-aware tokens such as E-4012."""

    def __init__(self, k1: float = 1.2, b: float = 0.75, stem: bool = True):
        self.k1 = k1
        self.b = b
        self._stemmer = Stemmer.Stemmer("english") if (stem and Stemmer) else None
        self._index: _Index | None = None
        self.documents: list = []

    @property
    def bm25(self) -> _Index | None:
        """The current index, or None before the first build."""
        return self._index

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        words = [w for w in _WORD.findall(text) if w not in _DROP]
        if self._stemmer:
            words = self._stemmer.stemWords(words)
        if "-" in text or "_" in text:
            words += _COMPOUND.findall(text)                  # codes like "e-4012", kept whole
        return words

    def index(self, chunks: list[dict]):
        """Build the index from in-memory chunks (dicts with id, content, metadata)."""
        if not chunks:
            self._index, self.documents = None, []
            return
        builder = _Builder(self)
        builder.add([c["content"] for c in chunks])
        ids = np.array([c.get("id", i) for i, c in enumerate(chunks)], dtype=np.int64)
        self._index = builder.build(ids, documents=list(chunks), store=None)
        self.documents = self._index.documents
        print(f"BM25 index built: {len(chunks)} documents")

    def load_from_vector_store(self, vector_store, batch_size: int = 5000) -> int:
        """Stream every chunk from the store; hits are fetched from the store at query time."""
        builder, ids = _Builder(self), []
        with vector_store.conn.cursor(name="bm25_load") as cur:
            cur.execute("SELECT id, content FROM chunks ORDER BY id")
            while rows := cur.fetchmany(batch_size):
                ids.extend(r[0] for r in rows)
                builder.add([r[1] for r in rows])
        vector_store.conn.commit()
        if not ids:
            self._index, self.documents = None, []
            return 0
        self._index = builder.build(np.array(ids, dtype=np.int64), documents=None,
                                    store=vector_store)
        self.documents = []
        print(f"BM25 index built: {len(ids)} documents")
        return len(ids)

    def search(self, query: str, top_k: int = 5) -> list[SparseResult]:
        """Return up to top_k chunks with a positive BM25 score, best first."""
        idx = self._index
        if idx is None:
            return []
        terms = [idx.vocab[t] for t in self._tokenize(query) if t in idx.vocab]
        if not terms:
            return []
        scores = np.zeros(idx.num_docs, dtype=np.float32)
        for t in terms:                                       # doc ids are unique within a term
            start, end = idx.indptr[t], idx.indptr[t + 1]
            scores[idx.doc_idx[start:end]] += idx.weights[start:end]
        k = min(top_k, int(np.count_nonzero(scores)))
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return self._results(idx, top, scores)

    def _results(self, idx: _Index, top: np.ndarray, scores: np.ndarray) -> list[SparseResult]:
        if idx.documents is not None:
            return [SparseResult(chunk_id=int(idx.chunk_ids[i]),
                                 content=idx.documents[i]["content"],
                                 metadata=idx.documents[i].get("metadata", {}) or {},
                                 score=float(scores[i])) for i in top]
        wanted = [int(idx.chunk_ids[i]) for i in top]
        with idx.store.conn.cursor() as cur:
            cur.execute("SELECT id, content, metadata FROM chunks WHERE id = ANY(%s)", (wanted,))
            rows = {r[0]: r for r in cur.fetchall()}
        return [SparseResult(chunk_id=cid, content=rows[cid][1], metadata=rows[cid][2] or {},
                             score=float(scores[i]))
                for i, cid in zip(top, wanted) if cid in rows]


class _Vocab(dict):
    """Term -> id map that assigns the next id to unseen terms."""

    def __missing__(self, term: str) -> int:
        self[term] = term_id = len(self)
        return term_id


class _Builder:
    """Accumulates (doc, term, tf) triples in batches, then computes BM25 weights in numpy."""

    def __init__(self, retriever: BM25Retriever):
        self.r = retriever
        self.vocab = _Vocab()
        self.lengths: list[int] = []
        self.parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def add(self, texts: list[str]):
        base, docs, terms = len(self.lengths), [], []
        lookup = self.vocab.__getitem__
        for i, text in enumerate(texts):
            tokens = self.r._tokenize(text)
            self.lengths.append(len(tokens))
            terms.extend(map(lookup, tokens))
            docs.extend([base + i] * len(tokens))
        if not terms:
            return
        keys = (np.asarray(docs, dtype=np.int64) << 32) | np.asarray(terms, dtype=np.int64)
        unique, tf = np.unique(keys, return_counts=True)
        self.parts.append(((unique >> 32).astype(np.int32),
                           (unique & 0xFFFFFFFF).astype(np.int32),
                           tf.astype(np.float32)))

    def build(self, chunk_ids: np.ndarray, documents: list | None, store) -> _Index:
        n_docs, n_terms = len(self.lengths), len(self.vocab)
        if self.parts:
            doc = np.concatenate([p[0] for p in self.parts])
            term = np.concatenate([p[1] for p in self.parts])
            tf = np.concatenate([p[2] for p in self.parts])
        else:
            doc = term = np.empty(0, dtype=np.int32)
            tf = np.empty(0, dtype=np.float32)
        self.parts = []

        lengths = np.asarray(self.lengths, dtype=np.float32)
        avg_len = float(lengths.mean()) if n_docs and lengths.mean() > 0 else 1.0
        df = np.bincount(term, minlength=n_terms).astype(np.float32)
        idf = np.log1p((n_docs - df + 0.5) / (df + 0.5))
        k1, b = self.r.k1, self.r.b
        weights = idf[term] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lengths[doc] / avg_len))

        order = np.argsort(term, kind="stable")                # group postings by term
        indptr = np.zeros(n_terms + 1, dtype=np.int64)
        np.cumsum(np.bincount(term, minlength=n_terms), out=indptr[1:])
        return _Index(vocab=dict(self.vocab), indptr=indptr, doc_idx=doc[order],
                      weights=weights[order].astype(np.float32), chunk_ids=chunk_ids,
                      num_docs=n_docs, documents=documents, store=store)