"""HNSW_EF_SEARCH: applied to every connection the store opens."""

import threading

import pytest

ZERO = "[" + ",".join(["0"] * 384) + "]"


def ef_search(conn) -> str:
    conn.execute("SELECT %s::vector <=> %s::vector", (ZERO, ZERO))   # loads pgvector settings
    value = conn.execute("SHOW hnsw.ef_search").fetchone()[0]
    conn.commit()
    return value


@pytest.mark.db
def test_setting_applies_to_every_thread_connection(clean_db, monkeypatch):
    from src.storage.vector_store import VectorStore

    monkeypatch.setenv("HNSW_EF_SEARCH", "200")
    store = VectorStore(connection_string=clean_db, embedding_dimension=384)
    try:
        assert store.ef_search == 200 and ef_search(store.conn) == "200"
        seen = []
        worker = threading.Thread(target=lambda: seen.append(ef_search(store.conn)))
        worker.start()
        worker.join()
        assert seen == ["200"]                                  # new thread, new connection
        store.search([0.0] * 384, top_k=1)                      # search works with it set
    finally:
        store.close()


@pytest.mark.db
def test_zero_keeps_pgvector_default_and_argument_wins(clean_db, monkeypatch):
    from src.storage.vector_store import VectorStore

    store = VectorStore(connection_string=clean_db, embedding_dimension=384)   # setting is 0
    try:
        assert store.ef_search == 0 and ef_search(store.conn) == "40"
    finally:
        store.close()
    monkeypatch.setenv("HNSW_EF_SEARCH", "200")
    store = VectorStore(connection_string=clean_db, embedding_dimension=384, ef_search=64)
    try:
        assert ef_search(store.conn) == "64"
    finally:
        store.close()


@pytest.mark.parametrize("value", [-1, 1001, "200"])
def test_invalid_values_are_rejected(value):
    from src.storage.vector_store import VectorStore

    with pytest.raises(ValueError, match="ef_search"):
        VectorStore(connection_string="postgresql://unused", ef_search=value)
