"""BEIR benchmark: IR metrics and an end-to-end run on a tiny corpus."""

import json
import sys

import psycopg
import pytest
from psycopg import sql

from src.evaluation.ir_metrics import mrr_at_k, ndcg_at_k, overlap_at_k, recall_at_k

CORPUS = [
    ("d1", "Refunds", "Refunds are issued within 14 days of a returned order."),
    ("d2", "Shipping", "Standard shipping takes five business days worldwide."),
    ("d3", "Passwords", "Reset your password from the login page using the email link."),
    ("d4", "Invoices", "Invoices can be downloaded as PDF from the billing dashboard."),
    ("d5", "Warranty", "The warranty covers manufacturing defects for two years."),
]
QUERIES = {"q1": "how long do refunds take", "q2": "reset password email link",
           "q3": "download invoice pdf billing"}
QRELS = [("q1", "d1", 1), ("q2", "d3", 1), ("q3", "d4", 2), ("q3", "d2", 0)]


def test_ndcg_perfect_and_graded():
    qrels = {"a": 2, "b": 1}
    assert ndcg_at_k(["a", "b", "c"], qrels, 10) == pytest.approx(1.0)
    swapped = (1 + 2 / 1.5849625) / (2 + 1 / 1.5849625)
    assert ndcg_at_k(["b", "a"], qrels, 10) == pytest.approx(swapped, rel=1e-6)
    assert ndcg_at_k(["c"], qrels, 10) == 0.0
    assert ndcg_at_k(["a"], {"a": 0}, 10) == 0.0         # no relevant docs


def test_recall_mrr_overlap():
    qrels = {"a": 1, "b": 1, "x": 0}
    assert recall_at_k(["c", "a"], qrels, 10) == 0.5
    assert recall_at_k(["c", "a"], qrels, 1) == 0.0
    assert mrr_at_k(["x", "c", "b"], qrels, 10) == pytest.approx(1 / 3)
    assert mrr_at_k(["c"], qrels, 10) == 0.0
    assert overlap_at_k(["a", "b"], ["b", "c"], 2) == 0.5


@pytest.fixture
def tiny_beir(tmp_path):
    folder = tmp_path / "beir" / "tiny_test"
    (folder / "qrels").mkdir(parents=True)
    with open(folder / "corpus.jsonl", "w") as f:
        for doc_id, title, text in CORPUS:
            f.write(json.dumps({"_id": doc_id, "title": title, "text": text}) + "\n")
    with open(folder / "queries.jsonl", "w") as f:
        for qid, text in QUERIES.items():
            f.write(json.dumps({"_id": qid, "text": text}) + "\n")
    with open(folder / "qrels" / "test.tsv", "w") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        f.writelines(f"{q}\t{d}\t{s}\n" for q, d, s in QRELS)
    return tmp_path


@pytest.mark.db
def test_benchmark_end_to_end(tiny_beir, db_url, fake_models, monkeypatch, capsys):
    from scripts import beir_benchmark

    monkeypatch.setenv("DATABASE_URL", db_url)
    out = tiny_beir / "out"
    argv = ["beir_benchmark", "--dataset", "tiny_test", "--data-dir", str(tiny_beir / "beir"),
            "--output-dir", str(out), "--batch", "2", "--sparse-weights", "1.0,0.5"]
    admin = psycopg.conninfo.make_conninfo(db_url, dbname="postgres")
    try:
        monkeypatch.setattr(sys, "argv", argv)
        beir_benchmark.main()
        first = json.loads(next(out.glob("*.json")).read_text())
        assert first["ingest"]["passages"] == 5 and not first["ingest"].get("skipped")
        assert first["queries"] == 3
        assert first["bm25_params"] == {"k1": 1.2, "b": 0.75}
        for name in ("dense", "bm25", "hybrid", "hybrid@0.5", "rerank"):
            m = first["configs"][name]
            assert m["ndcg@10"] == pytest.approx(1.0), name     # each query has an exact match
            assert m["mrr@10"] == pytest.approx(1.0), name
            assert m["latency_ms"]["n"] == 3
        assert first["configs"]["rerank"]["recall@100"] is None
        assert first["ann_recall@10"] == pytest.approx(1.0)
        assert first["sizes"]["hnsw_index_mb"] > 0

        for f in out.glob("*.json"):
            f.unlink()
        beir_benchmark.main()                                  # second run reuses the load
        second = json.loads(next(out.glob("*.json")).read_text())
        assert second["ingest"]["skipped"] is True
        assert "Corpus already loaded" in capsys.readouterr().out
    finally:
        with psycopg.connect(admin, autocommit=True) as conn:
            conn.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)")
                         .format(sql.Identifier("beir_tiny_test")))
