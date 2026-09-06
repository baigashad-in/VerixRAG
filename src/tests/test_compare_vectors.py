"""Compare precomputed passage vectors with local embeddings."""

import hashlib
import json

import numpy as np
import pytest

from scripts import compare_vectors as cv

TEXTS = [f"T{i}\npassage {i} text" for i in range(6)]


def _corpus(tmp_path):
    folder = tmp_path / "beir" / "tiny_cv"
    (folder / "qrels").mkdir(parents=True)
    with open(folder / "corpus.jsonl", "w") as f:
        for i in range(6):
            f.write(json.dumps({"_id": f"d{i}", "title": f"T{i}", "text": f"passage {i} text"}) + "\n")
    return folder


def _vectors():
    from src.tests.fakes import HashSentenceTransformer
    return np.asarray(HashSentenceTransformer().encode(TEXTS, normalize_embeddings=True), dtype=np.float32)


def test_compare_identical_scaled_and_flipped():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(20, 8)).astype(np.float32)
    assert cv.compare(a, a * 3)["min_cos"] > 0.99999                # scale does not matter
    b = a.copy()
    b[5] = -b[5]
    assert cv.compare(a, b)["min_cos"] < 0
    with pytest.raises(ValueError):
        cv.compare(a, a[:3])


def test_main_pass_fail_and_order(tmp_path, fake_models, capsys):
    folder = _corpus(tmp_path)
    good = _vectors()
    np.save(tmp_path / "good.npy", good)
    bad = good.copy()
    bad[2] = np.roll(bad[2], 7)
    np.save(tmp_path / "bad.npy", bad)
    base = ["--dataset", "tiny_cv", "--data-dir", str(folder.parent)]
    assert cv.main(base + ["--vectors", str(tmp_path / "good.npy")]) == 0
    assert cv.main(base + ["--vectors", str(tmp_path / "bad.npy")]) == 1
    assert cv.main(base + ["--vectors", str(tmp_path / "good.npy"), "--n", "3"]) == 0
    capsys.readouterr()

    # Manifest: right ids order passes, wrong model or order fails
    ids_ok = hashlib.sha256("\n".join(f"d{i}" for i in range(6)).encode()).hexdigest()
    man = tmp_path / "good.json"
    man.write_text(json.dumps({"model": "all-MiniLM-L6-v2", "count": 6, "ids_sha256": ids_ok}))
    assert cv.main(base + ["--vectors", str(tmp_path / "good.npy")]) == 0
    man.write_text(json.dumps({"model": "all-MiniLM-L6-v2", "count": 6, "ids_sha256": "x"}))
    assert cv.main(base + ["--vectors", str(tmp_path / "good.npy")]) == 1
    assert "passage order" in capsys.readouterr().out


def test_ids_hash_matches_manifest_formula(tmp_path):
    folder = _corpus(tmp_path)
    expected = hashlib.sha256("\n".join(["d0", "d1", "d2"]).encode()).hexdigest()
    assert cv.ids_sha256(folder, 3) == expected
