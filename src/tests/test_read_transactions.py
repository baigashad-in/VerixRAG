"""Read queries must not leave connections idle in transaction."""

import psycopg
import pytest
from psycopg.pq import TransactionStatus

from src.storage.vector_store import VectorStore


class FakeConn:
    """Minimal connection: a query opens a transaction, commit/rollback end it."""

    def __init__(self, status=TransactionStatus.IDLE, fail=False):
        self.info = type("Info", (), {"transaction_status": status})()
        self.fail, self.calls = fail, []

    def cursor(self):
        conn = self

        class Cur:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def execute(self, sql, params=None):
                if conn.info.transaction_status == TransactionStatus.IDLE:
                    conn.info.transaction_status = TransactionStatus.INTRANS
                if conn.fail:
                    conn.info.transaction_status = TransactionStatus.INERROR
                    raise psycopg.errors.UndefinedTable("boom")
        return Cur()

    def commit(self):
        self.calls.append("commit")
        self.info.transaction_status = TransactionStatus.IDLE

    def rollback(self):
        self.calls.append("rollback")
        self.info.transaction_status = TransactionStatus.IDLE


@pytest.fixture
def make_store():
    def make(conn):
        class Store(VectorStore):
            pass
        Store.conn = property(lambda self: conn)
        return Store.__new__(Store)
    return make


def test_read_ends_the_transaction_it_opened(make_store):
    conn = FakeConn()
    with make_store(conn).read_cursor() as cur:
        cur.execute("SELECT 1")
    assert conn.calls == ["commit"] and conn.info.transaction_status == TransactionStatus.IDLE


def test_failed_read_rolls_back(make_store):
    conn = FakeConn(fail=True)
    with pytest.raises(psycopg.errors.UndefinedTable):
        with make_store(conn).read_cursor() as cur:
            cur.execute("SELECT * FROM missing")
    assert conn.calls == ["rollback"] and conn.info.transaction_status == TransactionStatus.IDLE


def test_callers_transaction_is_left_open(make_store):
    conn = FakeConn(status=TransactionStatus.INTRANS)       # e.g. inside conn.transaction()
    with make_store(conn).read_cursor() as cur:
        cur.execute("SELECT 1")
    assert conn.calls == [] and conn.info.transaction_status == TransactionStatus.INTRANS


@pytest.mark.db
def test_real_reads_leave_connection_idle(clean_db, fake_models):
    from src.ingestion.embedding_service import EmbeddedChunk
    from src.retrieval.sparse_retrieval import BM25Retriever
    from src.tests import fakes

    _, _, embedder = fakes.build_engine([("x", {})])
    store = VectorStore(clean_db, embedding_dimension=fakes.DIM)
    try:
        store.insert_chunks([
            EmbeddedChunk(content=t, embedding=embedder.embed_text(t), metadata={"source": src})
            for t, src in (("Refunds take 14 days.", "a.md"), ("Shipping takes 5 days.", "b.md"))])
        idle = TransactionStatus.IDLE
        store.search(embedder.embed_text("refund"), top_k=2)
        assert store.conn.info.transaction_status == idle
        store.index_version(max_age_seconds=0)
        store.source_signatures()
        store.list_documents()
        store.domain_centroids()
        store.nearest_chunks(embedder.embed_text("refund"), k=1)
        assert store.conn.info.transaction_status == idle
        bm25 = BM25Retriever()
        bm25.load_from_vector_store(store)
        assert bm25.search("refund", top_k=1)
        assert store.conn.info.transaction_status == idle
        with store.conn.transaction():                  # caller-owned transaction survives
            store.search(embedder.embed_text("refund"), top_k=1)
            assert store.conn.info.transaction_status == TransactionStatus.INTRANS
        assert store.conn.info.transaction_status == idle
    finally:
        store.close()