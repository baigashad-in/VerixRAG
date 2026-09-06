"""Embed a BEIR corpus to a .npy file that beir_benchmark --embed-cache loads.

Reads passages like scripts/beir_benchmark.py (same text format and
order), embeds them with the app's model (normalised), and saves them in
pieces so an interrupted run resumes where it stopped. Writes a .json 
manifest (model, count, passage order hash) that 
scripts/compare_vectors.py checks.

Usage:
    python -m scripts.embed_corpus --dataset nq --limit 5000     # validation slice
    python -m scripts.embed_corpus --dataset nq                  # full corpus, resumable
    python -m scripts.beir_benchmark --dataset nq --embed-cache experiments/beir/vectors
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.beir_benchmark import download, embed_cache_path, iter_corpus


def count_passages(folder: Path) -> int:
    with open(folder / "corpus.jsonl") as f:
        return sum(1 for line in f if line.strip())


def manifest(dataset: str, embedder, count: int, ids_sha: str, seconds: float) -> dict:
    try:
        import sentence_transformers
        st_version = sentence_transformers.__version__
    except (ImportError, AttributeError):
        st_version = None
    return {"dataset": dataset, "model": embedder.model_name, "count": count,
            "dim": embedder.dimension, "ids_sha256": ids_sha, "normalized": True,
            "text_format": "title\\ntext, \\x00 removed, stripped (beir_benchmark.iter_corpus)",
            "device": str(getattr(embedder.model, "device", "unknown")),
            "sentence_transformers": st_version, "seconds": round(seconds, 1),
            "passages_per_s": round(count / seconds, 1) if seconds else None}


def encode(embedder, texts: list[str], batch: int) -> np.ndarray:
    return np.asarray(embedder.model.encode(texts, batch_size=batch, normalize_embeddings=True,
                                            show_progress_bar=False), dtype=np.float32)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Embed a BEIR corpus to a .npy vector file")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", default="experiments/beir")
    parser.add_argument("--out", default="experiments/beir/vectors")
    parser.add_argument("--piece", type=int, default=50_000, help="passages per saved piece")
    parser.add_argument("--batch", type=int, default=64, help="encode batch size (256 on a GPU)")
    parser.add_argument("--limit", type=int, help="embed only the first N passages (validation slice)")
    args = parser.parse_args(argv)

    from src.ingestion.embedding_service import EmbeddingService

    folder = download(args.dataset, Path(args.data_dir))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    embedder = EmbeddingService(model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    print(f"Device: {getattr(embedder.model, 'device', 'unknown')}", flush=True)

    if args.limit:                                       # slice: one file, no pieces
        block = next(iter_corpus(folder, args.limit))
        t0 = time.perf_counter()
        vectors = encode(embedder, [b[2] for b in block], args.batch)
        secs = time.perf_counter() - t0
        ids_sha = hashlib.sha256("\n".join(b[0] for b in block).encode()).hexdigest()
        path = out / f"{args.dataset}_slice{len(block)}.npy"
        np.save(path, vectors)
        path.with_suffix(".json").write_text(
            json.dumps(manifest(args.dataset, embedder, len(block), ids_sha, secs), indent=2))
        total = count_passages(folder)
        rate = len(block) / secs if secs else 0
        print(f"{len(block)} passages in {secs:.1f} s ({rate:,.0f}/s); "
              f"full corpus ({total:,}) estimate {total / rate / 3600:.1f} h" if rate else "")
        print(f"Saved {path}")
        return 0

    total = count_passages(folder)
    pieces_dir = out / "pieces" / args.dataset
    pieces_dir.mkdir(parents=True, exist_ok=True)
    ids_hash = hashlib.sha256()
    t0, embedded, n = time.perf_counter(), 0, 0
    for n, block in enumerate(iter_corpus(folder, args.piece)):
        ids_hash.update(("\n" if n else "").encode() + "\n".join(b[0] for b in block).encode())
        piece = pieces_dir / f"{n:05d}.npy"
        if piece.exists() and len(np.load(piece, mmap_mode="r")) == len(block):
            continue                                     # resume: already embedded
        np.save(piece, encode(embedder, [b[2] for b in block], args.batch))
        embedded += len(block)
        done = min((n + 1) * args.piece, total)
        rate = embedded / (time.perf_counter() - t0)
        print(f"  {done:,}/{total:,} passages ({rate:,.0f}/s, ~{(total - done) / rate / 60:.0f} min left)",
              flush=True)
    seconds = time.perf_counter() - t0

    final = embed_cache_path(str(out), folder, embedder.model_name)
    merged = np.lib.format.open_memmap(final, mode="w+", dtype=np.float32,
                                       shape=(total, embedder.dimension))
    row = 0
    for i in range(n + 1):
        part = np.load(pieces_dir / f"{i:05d}.npy")
        merged[row:row + len(part)] = part
        row += len(part)
    merged.flush()
    if row != total:
        raise RuntimeError(f"merged {row} rows, corpus has {total}")
    final.with_suffix(".json").write_text(
        json.dumps(manifest(args.dataset, embedder, total, ids_hash.hexdigest(), seconds), indent=2))
    print(f"Saved {final} ({final.stat().st_size / 2**30:.2f} GB); pieces in {pieces_dir} can be deleted")
    return 0


if __name__ == "__main__":
    sys.exit(main())