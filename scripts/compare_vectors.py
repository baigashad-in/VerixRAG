"""Check precomputed passage vectors (e.g. made on a GPU machine) against local ones.

Usage:
    python -m scripts.compare_vectors --dataset nq --vectors experiments/beir/vectors/nq_slice5000.npy
    python -m scripts.compare_vectors --dataset nq --vectors experiments/beir/vectors/nq__all-MiniLM-L6-v2.npy --n 5000
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.beir_benchmark import download, iter_corpus


def compare(a, b) -> dict:
    """Row-wise cosine similarity and largest absolute difference."""
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    cos = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    return {"n": len(a), "min_cos": round(float(cos.min()), 6), "mean_cos": round(float(cos.mean()), 6),
            "max_abs_diff": round(float(np.abs(a - b).max()), 6)}


def ids_sha256(folder: Path, n: int) -> str:
    """Hash of the first n passage ids, in corpus order."""
    ids = []
    with open(folder / "corpus.jsonl") as f:
        for line in f:
            if line.strip():
                ids.append(json.loads(line)["_id"])
                if len(ids) == n:
                    break
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compare external passage vectors with local ones")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--vectors", required=True, help=".npy file (manifest .json next to it is optional)")
    parser.add_argument("--data-dir", default="experiments/beir")
    parser.add_argument("--n", type=int, help="rows to compare (default: all rows, max 5000)")
    parser.add_argument("--min-cos", type=float, default=0.9999)
    args = parser.parse_args(argv)

    from src.ingestion.embedding_service import EmbeddingService

    path = Path(args.vectors)
    vectors = np.load(path, mmap_mode="r")
    n = min(args.n or 5000, len(vectors))
    folder = download(args.dataset, Path(args.data_dir))
    embedder = EmbeddingService(model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    problems = []

    manifest_path = path.with_suffix(".json")
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text())
        if m.get("model") != embedder.model_name:
            problems.append(f"model {m.get('model')!r} != local {embedder.model_name!r}")
        if m.get("count") != len(vectors):
            problems.append(f"manifest count {m.get('count')} != {len(vectors)} rows")
        if m.get("ids_sha256") != ids_sha256(folder, m.get("count", 0)):
            problems.append("passage order differs from the local corpus (ids hash)")
    else:
        print(f"No manifest at {manifest_path}; checking vectors only")

    block = next(iter_corpus(folder, n))
    local = embedder.model.encode([b[2] for b in block[:n]], batch_size=64,
                                  normalize_embeddings=True, show_progress_bar=False)
    stats = compare(vectors[:n], local)
    if stats["min_cos"] < args.min_cos:
        problems.append(f"min cosine {stats['min_cos']} < {args.min_cos}")

    print(f"{path}: {len(vectors):,} rows, compared first {n}: {stats}")
    for p in problems:
        print("  PROBLEM:", p)
    print("MATCH" if not problems else "MISMATCH")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
