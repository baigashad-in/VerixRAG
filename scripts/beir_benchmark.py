"""Benchmark retrieval at scale on a BEIR dataset, without LLM calls.

Loads the corpus into its own database (beir_<dataset>, never the app's
database), then reports quality against BEIR's human relevance labels,
query latency per configuration, ingest throughput and index size.

Usage:
    python -m scripts.beir_benchmark --dataset fiqa
    python -m scripts.beir_benchmark --dataset trec-covid --max-queries 200
    python -m scripts.beir_benchmark --dataset nq --configs dense --max-queries 500
"""

import argparse
import csv
import json
import os
import sys
import time
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg
from psycopg import sql

from src.evaluation.ir_metrics import mean, mrr_at_k, ndcg_at_k, overlap_at_k, recall_at_k
from src.observability.tracing import latency_stats, peak_rss_mb, rss_mb

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
ALL_CONFIGS = ("dense", "bm25", "hybrid", "rerank")
DEPTH = 100               # candidates scored for Recall@100
RERANK_TOP_K = 10         # production-style rerank (candidates: RERANK_CANDIDATES or 4 x top_k)

def download(dataset: str, data_dir: Path) -> Path:
    """Return the dataset folder, downloading and unzipping it if missing."""
    folder = data_dir / dataset
    if (folder / "corpus.jsonl").is_file():
        return folder
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / f"{dataset}.zip"
    print(f"Downloading {dataset}...", flush=True)
    urllib.request.urlretrieve(BEIR_URL.format(name=dataset), archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(data_dir)
    archive.unlink()
    return folder


def load_queries(folder: Path, split: str, limit: int | None) -> tuple[dict, dict]:
    qrels: dict[str, dict[str, int]] = {}
    with open(folder / "qrels" / f"{split}.tsv", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    queries = {}
    with open(folder / "queries.jsonl") as f:
        for line in f:
            q = json.loads(line)
            if q["_id"] in qrels:
                queries[q["_id"]] = q["text"]
    ids = sorted(queries)[:limit] if limit else sorted(queries)
    return {i: queries[i] for i in ids}, {i: qrels[i] for i in ids}


def iter_corpus(folder: Path, batch: int):
    block = []
    with open(folder / "corpus.jsonl") as f:
        for line in f:
            doc = json.loads(line)
            text = f"{doc.get('title', '')}\n{doc['text']}".replace("\x00", "").strip()
            block.append((doc["_id"], doc.get("title", ""), text))
            if len(block) == batch:
                yield block
                block = []
    if block:
        yield block


def database_url(dataset: str) -> str:
    base = os.environ["DATABASE_URL"]
    name = "beir_" + "".join(c if c.isalnum() else "_" for c in dataset.lower())
    with psycopg.connect(psycopg.conninfo.make_conninfo(base, dbname="postgres"),
                         autocommit=True) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone()
        if not exists:
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    return psycopg.conninfo.make_conninfo(base, dbname=name)


def ingest(store, embedder, folder: Path, batch: int, maintenance_mem: str) -> dict:
    """Bulk-load the corpus (COPY, index built afterwards). Skips if already loaded."""
    with open(folder / "corpus.jsonl") as f:
        total = sum(1 for line in f if line.strip())
    with store.conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM chunks")
        if cur.fetchone()[0] == total:
            print(f"Corpus already loaded ({total} passages)", flush=True)
            cur.execute("""CREATE INDEX IF NOT EXISTS chunks_embeddings_idx ON chunks
                           USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)""")
            store.conn.commit()
            return {"passages": total, "skipped": True}
        cur.execute("TRUNCATE chunks")
        cur.execute("DROP INDEX IF EXISTS chunks_embeddings_idx")
        cur.execute("DROP INDEX IF EXISTS chunks_metadata_idx")
    store.conn.commit()

    t0, embed_s, done = time.perf_counter(), 0.0, 0
    for block in iter_corpus(folder, batch):
        e0 = time.perf_counter()
        vectors = embedder.model.encode([b[2] for b in block], batch_size=64,
                                        normalize_embeddings=True, show_progress_bar=False)
        embed_s += time.perf_counter() - e0
        with store.conn.cursor() as cur, cur.copy(
                "COPY chunks (content, embedding, metadata) FROM STDIN") as copy:
            for (doc_id, title, text), vec in zip(block, vectors):
                copy.write_row((text, "[" + ",".join(f"{x:.6f}" for x in vec) + "]",
                                json.dumps({"source": doc_id, "title": title[:200]})))
        store.conn.commit()
        done += len(block)
        print(f"  {done}/{total} passages ({done / (time.perf_counter() - t0):.0f}/s)", flush=True)
    load_s = time.perf_counter() - t0

    i0 = time.perf_counter()
    with store.conn.cursor() as cur:
        cur.execute(sql.SQL("SET maintenance_work_mem = {}").format(sql.Literal(maintenance_mem)))
        cur.execute("""CREATE INDEX chunks_embeddings_idx ON chunks
                       USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)""")
        index_s = time.perf_counter() - i0
        cur.execute("CREATE INDEX chunks_metadata_idx ON chunks USING gin (metadata)")
        cur.execute("ANALYZE chunks")
    store.conn.commit()
    return {"passages": total, "load_s": round(load_s, 1), "embed_s": round(embed_s, 1),
            "passages_per_s": round(total / load_s, 1), "hnsw_build_s": round(index_s, 1)}


def sizes(store) -> dict:
    with store.conn.cursor() as cur:
        cur.execute("""SELECT pg_database_size(current_database()),
                              pg_total_relation_size('chunks'),
                              pg_relation_size('chunks_embeddings_idx')""")
        db, table, hnsw = cur.fetchone()
    return {"database_mb": round(db / 2**20, 2), "chunks_table_mb": round(table / 2**20, 2),
            "hnsw_index_mb": round(hnsw / 2**20, 2)}


class PrecomputedEmbedder:
    """Serve query embeddings computed up front, so latency covers retrieval only."""

    def __init__(self, vectors: dict, fallback):
        self.vectors, self.fallback = vectors, fallback

    def embed_text(self, text: str) -> list[float]:
        return self.vectors.get(text) or self.fallback.embed_text(text)


def doc_ids(results) -> list[str]:
    ids = []
    for r in results:
        meta = r.get("metadata") if isinstance(r, dict) else getattr(r, "metadata", None)
        ids.append((meta or {}).get("source"))
    return ids


def main():
    parser = argparse.ArgumentParser(description="BEIR retrieval benchmark (no LLM calls)")
    parser.add_argument("--dataset", required=True, help="BEIR name, e.g. fiqa, trec-covid, nq")
    parser.add_argument("--split", default="test")
    parser.add_argument("--configs", default=",".join(ALL_CONFIGS))
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--data-dir", default="experiments/beir")
    parser.add_argument("--batch", type=int, default=2000, help="passages per embed/COPY block")
    parser.add_argument("--ef-search", type=int, default=200)
    parser.add_argument("--ann-sample", type=int, default=100, help="queries for HNSW vs exact")
    parser.add_argument("--maintenance-mem", default="1GB")
    parser.add_argument("--output-dir", default="experiments")
    parser.add_argument("--sparse-weights", default="1.0",
                        help="hybrid BM25 weights to sweep, e.g. 1.0,0.5,0.3")
    parser.add_argument("--rerank-weights", default="1.0",
                        help="BM25 weights for the candidates the reranker scores")
    parser.add_argument("--rerank-candidates", default="",
                        help="candidate counts to sweep, e.g. 40,20,10 (default: setting)")
    parser.add_argument("--rerank-max-lengths", default="",
                        help="reranker token limits to sweep, e.g. 512,256,128 (default: setting)")
    parser.add_argument("--bm25-k1", type=float, default=1.2)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    args = parser.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    if set(configs) - set(ALL_CONFIGS):
        parser.error(f"configs must be from {ALL_CONFIGS}")

    from src.ingestion.embedding_service import EmbeddingService
    from src.retrieval.retrieval_engine import RetrievalEngine
    from src.retrieval.sparse_retrieval import BM25Retriever
    from src.storage.vector_store import VectorStore

    folder = download(args.dataset, Path(args.data_dir))
    queries, qrels = load_queries(folder, args.split, args.max_queries)
    embedder = EmbeddingService(model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    store = VectorStore(connection_string=database_url(args.dataset),
                        embedding_dimension=embedder.dimension)
    report = {"dataset": args.dataset, "split": args.split, "queries": len(queries),
              "ef_search": args.ef_search,
              "timestamp": datetime.now().isoformat(), "configs": {}}
    report["ingest"] = ingest(store, embedder, folder, args.batch, args.maintenance_mem)
    report["sizes"] = sizes(store)
    store.conn.execute(sql.SQL("SET hnsw.ef_search = {}").format(sql.Literal(args.ef_search)))
    store.conn.commit()

    bm25 = None
    if {"bm25", "hybrid", "rerank"} & set(configs):
        b0, rss0 = time.perf_counter(), rss_mb()
        bm25 = BM25Retriever(k1=args.bm25_k1, b=args.bm25_b)
        bm25.load_from_vector_store(store)
        report["bm25_build"] = {"s": round(time.perf_counter() - b0, 1),
                                "rss_growth_mb": round(rss_mb() - rss0, 1)}
    reranker = None
    if "rerank" in configs:
        from src.retrieval.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker(model_name=os.getenv(
            "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    engine = RetrievalEngine(vector_store=store, bm25_retriever=bm25, embedding_service=embedder,
                             reranker=reranker, domain_scorer=None)
    engine.cache = None
    embeddings, embed_times = {}, []
    for qid, text in queries.items():
        t0 = time.perf_counter()
        embeddings[qid] = embedder.embed_text(text)
        embed_times.append((time.perf_counter() - t0) * 1000)
    report["query_embed_ms"] = latency_stats(embed_times)
    by_text = {queries[qid]: emb for qid, emb in embeddings.items()}
    engine.embedder = PrecomputedEmbedder(by_text, embedder)   # time retrieval, not embedding

    runners = {
        "dense": lambda q, e: store.search(e, top_k=DEPTH),
        "bm25": lambda q, e: bm25.search(q, top_k=DEPTH),
        "hybrid": lambda q, e: engine.hybrid.search(q, e, top_k=DEPTH, fetch_k=DEPTH),
        "rerank": lambda q, e: engine._retrieve_uncached(q, RERANK_TOP_K, True, None),
    }
    def values(raw, cast):
        return [cast(v) for v in raw.split(",") if v.strip()] or [None]

    sweeps = {"hybrid": args.sparse_weights, "rerank": args.rerank_weights}
    runs = []                                   # (label, runner, weight, candidates, max length)
    for name in configs:
        weights = values(sweeps.get(name, "1.0"), float)
        extra = ([(c, n) for c in values(args.rerank_candidates, int)
                  for n in values(args.rerank_max_lengths, int)]
                 if name == "rerank" else [(None, None)])
        for w in weights:
            for c, n in extra:
                label = name + ("" if w == 1.0 else f"@{w:g}") + \
                        ("" if c is None else f" c{c}") + ("" if n is None else f" len{n}")
                runs.append((label, runners[name], w, c, n))
    report["bm25_params"] = {"k1": args.bm25_k1, "b": args.bm25_b}
    base_candidates = engine.rerank_candidates
    base_length = reranker.max_length if reranker else None
    report["rerank_defaults"] = {"candidates": base_candidates or 4 * RERANK_TOP_K,
                                 "max_length": base_length}
    for name, runner, weight, candidates, max_length in runs:
        engine.hybrid.sparse_weight = weight
        engine.rerank_candidates = base_candidates if candidates is None else candidates
        if reranker and (max_length or base_length):
            reranker.max_length = max_length or base_length
        print(f"Running {name} on {len(queries)} queries...", flush=True)
        scores = {"ndcg@10": [], "recall@10": [], "recall@100": [], "mrr@10": []}
        times = []
        for qid, text in queries.items():
            t0 = time.perf_counter()
            ranked = doc_ids(runner(text, embeddings[qid]))
            times.append((time.perf_counter() - t0) * 1000)
            scores["ndcg@10"].append(ndcg_at_k(ranked, qrels[qid], 10))
            scores["recall@10"].append(recall_at_k(ranked, qrels[qid], 10))
            scores["mrr@10"].append(mrr_at_k(ranked, qrels[qid], 10))
            if not name.startswith("rerank"):
                scores["recall@100"].append(recall_at_k(ranked, qrels[qid], 100))
        report["configs"][name] = {**{k: mean(v) for k, v in scores.items()},
                                   "latency_ms": latency_stats(times)}

    # HNSW (approximate) vs exact search on a sample of queries
    sample = list(embeddings.items())[:args.ann_sample]
    if "dense" in configs and sample:
        approx = [doc_ids(store.search(emb, top_k=10)) for _, emb in sample]
        store.conn.commit()
        with store.conn.transaction():
            store.conn.execute("SET LOCAL enable_indexscan = off")     # exact scan
            exact = [doc_ids(store.search(emb, top_k=10)) for _, emb in sample]
        report["ann_recall@10"] = mean([overlap_at_k(a, e, 10) for a, e in zip(approx, exact)])
    report["memory_mb"] = {"rss": rss_mb(), "peak": peak_rss_mb()}

    print(f"\n{args.dataset}: {report['ingest']['passages']} passages, {len(queries)} queries")
    print(f"{'config':<22}{'nDCG@10':>9}{'R@10':>8}{'R@100':>8}{'MRR@10':>8}{'p50 ms':>9}{'p95 ms':>9}")
    for name, m in report["configs"].items():
        lat = m["latency_ms"]
        print(f"{name:<22}{m['ndcg@10']:>9}{m['recall@10']:>8}{'-' if m['recall@100'] is None else m['recall@100']:>8}"
              f"{m['mrr@10']:>8}{lat['p50']:>9}{lat['p95']:>9}")
    print(f"\ningest: {report['ingest']}\nsizes: {report['sizes']}")
    if "bm25_build" in report:
        print(f"bm25 build {report['bm25_params']}: {report['bm25_build']}")
    if "ann_recall@10" in report:
        print(f"HNSW recall@10 vs exact search (ef_search={args.ef_search}): {report['ann_recall@10']}")
    if "rerank_defaults" in report and reranker:
        print(f"rerank defaults: {report['rerank_defaults']}")
    print(f"query embedding (not in the latencies above): {report['query_embed_ms']}")
    print(f"memory: {report['memory_mb']}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(args.output_dir) / f"beir_{args.dataset}_{report['timestamp'][:19].replace(':', '-')}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"Saved: {path}")
    store.close()


if __name__ == "__main__":
    main()
