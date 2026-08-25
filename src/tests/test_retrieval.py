"""Tests for retrieval components."""

import json
import math

import pytest
import pytest
from src.retrieval.sparse_retrieval import BM25Retriever

class TestBM25Retriever:

    @pytest.fixture
    def indexed_retriever(self):
        retriever = BM25Retriever()
        chunks = [
            {"id": 1, "content": "Refund policy allows returns within 30 days", "metadata": {}},
            {"id": 2, "content": "Express shipping costs fifteen dollars", "metadata": {}},
            {"id": 3, "content": "Account deletion removes all personal data", "metadata": {}},
            {"id": 4, "content":"Store credit never expires after refund", "metadata": {}},
            {"id": 5, "content": "Password must contain eight characters minimum", "metadata": {}},
            {"id": 6, "content": "Two factor authentication enhances security", "metadata": {}},
            {"id": 7, "content": "Order tracking is available for all shipments", "metadata": {}},
        ]
        retriever.index(chunks)
        return retriever
    
    def test_index_builds(self, indexed_retriever):
        assert indexed_retriever.bm25 is not None
        assert len(indexed_retriever.documents) == 7

    def test_search_returns_results(self, indexed_retriever):
        results = indexed_retriever.search("refund policy", top_k=2)
        assert len(results) > 0
        assert len(results) <= 2

    def test_relevant_result_first(self, indexed_retriever):
        results = indexed_retriever.search("refund", top_k=3)
        # Chunks about refund should score higher
        contents = [r.content for r in results]
        assert any("refund" in c.lower() for c in contents)

    def test_search_before_index_returns_empty(self):
        # Deliberate: with an empty knowledge base (first start, before
        # ingestion) hybrid search must still work on dense results alone
        # instead of crashing, so an unindexed BM25 returns no results.
        retriever = BM25Retriever()
        assert retriever.search("test query") == []

    def test_no_results_for_unrelated_query(self, indexed_retriever):
        results = indexed_retriever.search("quantum physics", top_k=3)
        # Should return empty or very low scores
        for r in results:
            assert r.score < 1.0

CORPUS = [
    {"id": 10, "content": "Refunds are processed within 14 days of the return.", "metadata": {"source": "a"}},
    {"id": 11, "content": "Error E-4012 means the payment card was declined.", "metadata": {"source": "b"}},
    {"id": 12, "content": "Shipping refunds are not available for express orders.", "metadata": {"source": "c"}},
    {"id": 13, "content": "Reset the router, then wait two minutes before reconnecting.", "metadata": {"source": "d"}},
]


def reference_scores(retriever, chunks, query):
    """Textbook Lucene BM25, computed document by document."""
    docs = [retriever._tokenize(c["content"]) for c in chunks]
    avg = sum(map(len, docs)) / len(docs)
    k1, b, n = retriever.k1, retriever.b, len(docs)
    scores = []
    for tokens in docs:
        s = 0.0
        for t in retriever._tokenize(query):
            df = sum(t in d for d in docs)
            tf = tokens.count(t)
            if tf:
                idf = math.log1p((n - df + 0.5) / (df + 0.5))
                s += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(tokens) / avg))
        scores.append(s)
    return scores


class TestBM25Scoring:

    def test_matches_reference_formula(self):
        retriever = BM25Retriever()
        retriever.index(CORPUS)
        query = "refunds for express shipping"
        expected = reference_scores(retriever, CORPUS, query)
        got = {r.chunk_id: r.score for r in retriever.search(query, top_k=10)}
        for chunk, want in zip(CORPUS, expected):
            assert got.get(chunk["id"], 0.0) == pytest.approx(want, rel=1e-5)

    def test_stemming_matches_word_forms(self):
        retriever = BM25Retriever()
        retriever.index(CORPUS)
        assert retriever.search("reconnect", top_k=1)[0].chunk_id == 13
        assert retriever.search("refund", top_k=1)[0].chunk_id in (10, 12)

    def test_stopwords_only_query_returns_nothing(self):
        retriever = BM25Retriever()
        retriever.index(CORPUS)
        assert retriever.search("the of and is", top_k=5) == []

    def test_error_codes_match_whole_and_by_number(self):
        retriever = BM25Retriever()
        retriever.index(CORPUS)
        assert retriever.search("what does E-4012 mean", top_k=1)[0].chunk_id == 11
        assert retriever.search("4012", top_k=1)[0].chunk_id == 11

    def test_metadata_returned_and_top_k_respected(self):
        retriever = BM25Retriever()
        retriever.index(CORPUS)
        results = retriever.search("refunds", top_k=1)
        assert len(results) == 1 and results[0].metadata["source"] in ("a", "c")

    def test_rebuild_replaces_index(self):
        retriever = BM25Retriever()
        retriever.index(CORPUS)
        retriever.index([{"id": 99, "content": "Completely different router text", "metadata": {}}])
        assert [r.chunk_id for r in retriever.search("router", top_k=5)] == [99]
        assert retriever.search("refunds", top_k=5) == []


@pytest.mark.db
def test_load_from_vector_store_fetches_hits(clean_db):
    from src.storage.vector_store import VectorStore

    store = VectorStore(connection_string=clean_db, embedding_dimension=384)
    zero = "[" + ",".join(["0"] * 384) + "]"
    with store.conn.cursor() as cur:
        for c in CORPUS:
            cur.execute("INSERT INTO chunks (id, content, embedding, metadata) "
                        "VALUES (%s, %s, %s::vector, %s::jsonb)",
                        (c["id"], c["content"], zero, json.dumps(c["metadata"])))
    store.conn.commit()

    retriever = BM25Retriever()
    assert retriever.load_from_vector_store(store, batch_size=3) == 4
    assert retriever.documents == []                       # contents stay in Postgres
    in_memory = BM25Retriever()
    in_memory.index(CORPUS)
    for query in ("refunds express", "E-4012", "router reconnect"):
        from_db = retriever.search(query, top_k=3)
        assert [(r.chunk_id, round(r.score, 5)) for r in from_db] == \
               [(r.chunk_id, round(r.score, 5)) for r in in_memory.search(query, top_k=3)]
        assert from_db[0].content and from_db[0].metadata["source"]
    store.close()