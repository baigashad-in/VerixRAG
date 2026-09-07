"""Saved BM25 index: reused while chunks and settings are unchanged, rebuilt otherwise."""

import json

import numpy as np
import pytest

from src.retrieval.sparse_retrieval import BM25Retriever

CORPUS = [
    (1, "Refunds are issued within 14 days.", "a.md"),
    (2, "Express shipping costs fifteen dollars.", "b.md"),
    (3, "Error E-4012 means the router lost its connection.", "c.md"),
    (4, "Store credit never expires after a refund.", "a.md"),
]
QUERIES = ("refunds", "E-4012", "express shipping", "router connection")
SIGNATURE = {"format": 1, "database": "unit_test", "table": "1:1", "count": 2, "max_id": 2, "id_sum": 3,
             "k1": 1.2, "b": 0.75, "stem": True}


def ranked(retriever):
    return [[(r.chunk_id, round(r.score, 5)) for r in retriever.search(q, top_k=3)]
            for q in QUERIES]


def test_saved_arrays_and_vocab_round_trip(tmp_path):
    built = BM25Retriever()
    built.index([{"id": 10, "content": "Refunds take 14 days", "metadata": {}},
                 {"id": 20, "content": "Error E-4012 on the router", "metadata": {}}])
    built._save_cache(tmp_path, SIGNATURE)
    loaded = BM25Retriever()
    assert loaded._load_cache(tmp_path, SIGNATURE, store=None)
    a, b = built.bm25, loaded.bm25
    assert a.vocab == b.vocab and a.num_docs == b.num_docs
    for name in ("indptr", "doc_idx", "weights", "chunk_ids"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name


def test_partial_or_mismatched_cache_is_not_loaded(tmp_path, capsys):
    built = BM25Retriever()
    built.index([{"id": 1, "content": "Refunds take 14 days", "metadata": {}}])
    built._save_cache(tmp_path, SIGNATURE)
    folder = tmp_path / SIGNATURE["database"]
    assert not BM25Retriever()._load_cache(tmp_path, {**SIGNATURE, "count": 3}, None)
    assert "stale" in capsys.readouterr().out
    (folder / "vocab.txt").write_text("refund", encoding="utf-8")       # one term missing
    assert not BM25Retriever()._load_cache(tmp_path, SIGNATURE, None)
    assert "inconsistent" in capsys.readouterr().out
    (folder / "weights.npy").write_bytes(b"not a numpy file")
    assert not BM25Retriever()._load_cache(tmp_path, SIGNATURE, None)
    (folder / "meta.json").unlink()                                      # save never finished
    assert not BM25Retriever()._load_cache(tmp_path, SIGNATURE, None)


@pytest.fixture
def store(clean_db):
    from src.storage.vector_store import VectorStore

    store = VectorStore(connection_string=clean_db, embedding_dimension=384)
    for chunk_id, content, source in CORPUS:
        insert(store, chunk_id, content, source)
    yield store
    store.close()


def insert(store, chunk_id, content, source):
    zero = "[" + ",".join(["0"] * 384) + "]"
    with store.conn.cursor() as cur:
        cur.execute("INSERT INTO chunks (id, content, embedding, metadata) "
                    "VALUES (%s, %s, %s::vector, %s::jsonb)",
                    (chunk_id, content, zero, json.dumps({"source": source})))
    store.conn.commit()
    store.invalidate_version()


@pytest.mark.db
def test_second_load_uses_cache_with_same_results(store, tmp_path, monkeypatch):
    first = BM25Retriever()
    assert first.load_from_vector_store(store, cache_dir=str(tmp_path)) == 4
    assert first.load_source == "built"
    assert (tmp_path / store.conn.info.dbname / "meta.json").exists()

    import src.retrieval.sparse_retrieval as sr

    def no_build(*args, **kwargs):
        raise AssertionError("index was rebuilt instead of loaded")
    monkeypatch.setattr(sr, "_Builder", no_build)
    second = BM25Retriever()
    assert second.load_from_vector_store(store, cache_dir=str(tmp_path)) == 4
    assert second.load_source == "cache"
    assert ranked(second) == ranked(first)
    assert second.search("refunds", top_k=1)[0].metadata["source"] == "a.md"


@pytest.mark.db
def test_changed_chunks_or_settings_rebuild(store, tmp_path):
    BM25Retriever().load_from_vector_store(store, cache_dir=str(tmp_path))
    insert(store, 5, "The warranty covers routers for two years.", "d.md")
    after_insert = BM25Retriever()
    after_insert.load_from_vector_store(store, cache_dir=str(tmp_path))
    assert after_insert.load_source == "built"
    assert 5 in [r.chunk_id for r in after_insert.search("warranty", top_k=3)]

    other_k1 = BM25Retriever(k1=0.9)
    other_k1.load_from_vector_store(store, cache_dir=str(tmp_path))
    assert other_k1.load_source == "built"
    reloaded = BM25Retriever(k1=0.9)
    reloaded.load_from_vector_store(store, cache_dir=str(tmp_path))
    assert reloaded.load_source == "cache" and ranked(reloaded) == ranked(other_k1)


@pytest.mark.db
def test_recreated_table_with_same_ids_rebuilds(store, tmp_path):
    BM25Retriever().load_from_vector_store(store, cache_dir=str(tmp_path))
    with store.conn.cursor() as cur:
        cur.execute("CREATE TABLE chunks_copy AS SELECT * FROM chunks")
        cur.execute("TRUNCATE chunks")
        cur.execute("INSERT INTO chunks SELECT * FROM chunks_copy")
        cur.execute("UPDATE chunks SET content = 'Gift cards are sold online.' WHERE id = 1")
        cur.execute("DROP TABLE chunks_copy")
    store.conn.commit()
    retriever = BM25Retriever()
    retriever.load_from_vector_store(store, cache_dir=str(tmp_path))
    assert retriever.load_source == "built"                 # same ids, new table file
    assert [r.chunk_id for r in retriever.search("gift cards", top_k=1)] == [1]


@pytest.mark.db
def test_without_cache_dir_nothing_is_saved(store, tmp_path):
    retriever = BM25Retriever()
    retriever.load_from_vector_store(store)
    assert retriever.load_source == "built" and not list(tmp_path.iterdir())
