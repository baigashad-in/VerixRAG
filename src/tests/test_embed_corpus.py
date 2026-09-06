"""Embedding a BEIR corpus to the file beir_benchmark --embed-cache loads."""

import json
import sys

import numpy as np
import psycopg
import pytest
from psycopg import sql

from scripts import compare_vectors, embed_corpus
from src.tests.test_beir_benchmark import CORPUS, tiny_beir  # noqa: F401 (fixture)


def _args(tiny_beir, *extra):
    return ["--dataset", "tiny_test", "--data-dir", str(tiny_beir / "beir"),
            "--out", str(tiny_beir / "vectors"), *extra]


def test_full_run_writes_file_and_manifest_and_resumes(tiny_beir, fake_models, capsys):
    assert embed_corpus.main(_args(tiny_beir, "--piece", "2")) == 0
    final = tiny_beir / "vectors" / "tiny_test__all-MiniLM-L6-v2.npy"
    vec = np.load(final)
    assert vec.shape == (len(CORPUS), 384)
    assert np.allclose(np.linalg.norm(vec, axis=1), 1.0, atol=1e-5)
    m = json.loads(final.with_suffix(".json").read_text())
    folder = tiny_beir / "beir" / "tiny_test"
    assert m["count"] == len(CORPUS) and m["ids_sha256"] == compare_vectors.ids_sha256(folder, len(CORPUS))
    assert len(list((tiny_beir / "vectors" / "pieces" / "tiny_test").glob("*.npy"))) == 3

    capsys.readouterr()
    assert embed_corpus.main(_args(tiny_beir, "--piece", "2")) == 0       # all pieces exist
    assert "passages (" not in capsys.readouterr().out                   # nothing re-embedded
    assert np.array_equal(np.load(final), vec)

    vectors = ["--dataset", "tiny_test", "--data-dir", str(tiny_beir / "beir"), "--vectors", str(final)]
    assert compare_vectors.main(vectors) == 0                             # same text, order, model


def test_slice_mode(tiny_beir, fake_models):
    assert embed_corpus.main(_args(tiny_beir, "--limit", "3")) == 0
    path = tiny_beir / "vectors" / "tiny_test_slice3.npy"
    assert np.load(path).shape == (3, 384)
    assert json.loads(path.with_suffix(".json").read_text())["count"] == 3


@pytest.mark.db
def test_benchmark_loads_the_embedded_file(tiny_beir, db_url, fake_models, monkeypatch):
    from scripts import beir_benchmark

    monkeypatch.setenv("DATABASE_URL", db_url)
    assert embed_corpus.main(_args(tiny_beir)) == 0
    out = tiny_beir / "out"
    monkeypatch.setattr(sys, "argv", ["beir_benchmark", "--dataset", "tiny_test",
                                      "--data-dir", str(tiny_beir / "beir"), "--output-dir", str(out),
                                      "--configs", "dense", "--embed-cache", str(tiny_beir / "vectors")])
    admin = psycopg.conninfo.make_conninfo(db_url, dbname="postgres")
    try:
        beir_benchmark.main()
        report = json.loads(next(out.glob("*.json")).read_text())
        assert report["ingest"]["embed_cache"] == "hit"
        assert report["configs"]["dense"]["ndcg@10"] == pytest.approx(1.0)
    finally:
        with psycopg.connect(admin, autocommit=True) as conn:
            conn.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)")
                         .format(sql.Identifier("beir_tiny_test")))
