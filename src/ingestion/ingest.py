"""
The complete ingestion pipeline:
Document -> Chunks -> Metadata -> Embeddings -> Dedup -> Vector Store

1. Load, and compute an ingest signature (content hash + settings).
   If the stored signature matches, the document is skipped: no
   re-chunking, no re-embedding. Restarts become near-instant.
2. Chunk, then attach metadata to every chunk: title, section path
   (e.g. "Account Management Guide › Password Reset"), character
   offset, document hash (its version) and ingestion time.
3. Embed all changed documents in one batch.
4. Drop exact and near-duplicate chunks, both within this batch and
   against chunks already stored from other documents.
5. Replace each document's chunks in a single transaction.
6. Optionally prune documents that no longer exist on disk.

This script is run when new documents are added to the knowledge base.
"""

from src.ingestion.document_loader import DocumentLoader, Document
from src.ingestion.embedding_service import EmbeddingService
from src.storage.vector_store import VectorStore
from src.ingestion.chunker_factory import ChunkerFactory
from src.ingestion.dedup import Deduplicator
from src.ingestion.metadata import (
    ChunkLocator, content_hash, document_title, ingest_signature, parse_front_matter, utc_now,
)
from src.observability.tracing import measure, peak_rss_mb, rss_mb
from src.retrieval.scoping import load_domain_config


class IngestionPipeline:
    """Orchestrates the full ingestion flow.

    Designed so that one can swap any component:
    - Different chunker? Change one parameter.
    - Different embedding model? Change one parameter.
    - Different vector DB? Implement the same interface.
    """

    def __init__(self, chunk_strategy: str = "recursive",
                 chunk_size: int = 512,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 db_connection: str = "postgresql://user:pass@localhost:5432/ragdb",
                 documents_root: str = "./documents"):
        from src.config import get_settings
        settings = get_settings()

        self.loader = DocumentLoader(allowed_directory=documents_root)
        self.chunker = ChunkerFactory.create(
            strategy=chunk_strategy,
            chunk_size=chunk_size,
        )
        self.embedder = EmbeddingService(model_name=embedding_model)
        self.store = VectorStore(
            connection_string=db_connection,
            embedding_dimension=self.embedder.dimension,
        )
        self.deduplicator = (Deduplicator(settings.dedup_cosine, settings.dedup_jaccard)
                             if settings.dedup_enabled else None)
        self.domains = load_domain_config(settings.domains_config)

        # Log the configuration - critical for experiment tracking.
        # Anything that changes the stored chunks belongs in here, because
        # it is part of each document's ingest signature.
        self.config = {
            "chunk_strategy": chunk_strategy,
            "chunk_size": chunk_size,
            "embedding_model": embedding_model,
            "dedup": ([settings.dedup_cosine, settings.dedup_jaccard]
                      if settings.dedup_enabled else None),
            "metadata_version": 2,
        }
        self.last_perf: dict = {}
        self.last_dedup: list[dict] = []
        print(f"Pipeline configured: {self.config}")

    # -- per-document preparation --

    def _domain(self, doc: Document) -> str | None:
        front, _ = parse_front_matter(doc.content)
        return self.domains.domain_for(doc.metadata["source"], front)

    def _doc_config(self, doc: Document) -> dict:
        """Settings that affect this document's chunks.

        Includes the document's domain, so remapping a file in
        config/domains.json re-ingests just that file.
        """
        return {**self.config, "domain": self._domain(doc)}

    def _prepare(self, doc: Document) -> list:
        """Chunk a document and attach rich metadata to every chunk."""
        front, body = parse_front_matter(doc.content)
        if front:
            doc = Document(content=body, metadata={**doc.metadata, **front})

        source = doc.metadata["source"]
        meta = {
            "doc_hash": content_hash(doc.content),
            "ingest_sig": doc.metadata["ingest_sig"],
            "title": doc.metadata.get("title") or document_title(doc.content, source),
            "ingested_at": utc_now(),
        }
        domain = self.domains.domain_for(source, front)
        if domain:
            meta["domain"] = domain
        chunks = self.chunker.chunk(doc)
        locator = ChunkLocator(doc.content)
        for chunk in chunks:
            position, section = locator.locate(chunk.content)
            chunk.metadata.update(meta)
            chunk.metadata["section"] = section
            if position is not None:
                chunk.metadata["char_start"] = position
            chunk.metadata.pop("char_count", None)   # document-level, not chunk-level
        return chunks

    # -- main entry points --

    def ingest_file(self, file_path: str, force: bool = False) -> dict:
        """Ingest a single file end-to-end."""
        document = self.loader.load_file(file_path)
        results = self.ingest_documents([document], force=force)
        return results[0] if results else {"file": file_path, "status": "unchanged"}

    def ingest_directory(self, dir_path: str, force: bool = False,
                         prune: bool = False) -> list[dict]:
        """Ingest all documents in a directory.

        Args:
            force: re-ingest even if a document's signature is unchanged
            prune: delete stored documents that are no longer in dir_path
        """
        perf = {"rss_start_mb": rss_mb()}
        with measure("load", perf):
            documents = self.loader.load_directory(dir_path)
        return self.ingest_documents(documents, force=force, prune=prune,
                                     perf=perf, load_ms=perf.pop("load")["ms"])

    def ingest_documents(self, documents: list[Document], force: bool = False,
                         prune: bool = False, perf: dict | None = None,
                         load_ms: float = 0.0) -> list[dict]:
        perf = perf if perf is not None else {"rss_start_mb": rss_mb()}
        stage_ms = {"load": load_ms, "signature": 0.0, "chunk": 0.0,
                    "embed": 0.0, "dedup": 0.0, "store": 0.0, "prune": 0.0}
        results: list[dict] = []

        # 1. Which documents changed?
        timer: dict = {}
        with measure("signature", timer):
            stored = self.store.source_signatures()
            changed = []
            for doc in documents:
                source = doc.metadata["source"]
                doc.metadata["ingest_sig"] = ingest_signature(doc.content, self._doc_config(doc))
                if not force and stored.get(source) == doc.metadata["ingest_sig"]:
                    results.append({"file": source, "status": "unchanged", "chunks_created": 0})
                else:
                    changed.append(doc)
        stage_ms["signature"] = timer["signature"]["ms"]

        # 2. Chunk + metadata
        prepared: list[tuple[Document, list]] = []
        with measure("chunk", timer):
            for doc in changed:
                prepared.append((doc, self._prepare(doc)))
        stage_ms["chunk"] = timer["chunk"]["ms"]

        # 3. Embed every changed chunk in one batch
        all_chunks = [c for _, chunks in prepared for c in chunks]
        with measure("embed", timer):
            embedded = self.embedder.embed_chunks(all_chunks) if all_chunks else []
        stage_ms["embed"] = timer["embed"]["ms"]

        # 4. Deduplicate: within the batch, and against other stored docs
        changed_sources = [d.metadata["source"] for d in changed]
        with measure("dedup", timer):
            if self.deduplicator and embedded:
                lookup = lambda emb: self.store.nearest_chunks(  # noqa: E731
                    emb, k=3, exclude_sources=changed_sources)
                report = self.deduplicator.filter(embedded, external_lookup=lookup)
                kept, self.last_dedup = report.kept, report.dropped
            else:
                kept, self.last_dedup = embedded, []
        stage_ms["dedup"] = timer["dedup"]["ms"]
        dropped_by_source: dict[str, int] = {}
        for d in self.last_dedup:
            dropped_by_source[d["source"]] = dropped_by_source.get(d["source"], 0) + 1
            print(f"  dedup: dropped {d['chunk']} ({d['reason']} duplicate of {d['duplicate_of']})")

        # 5. Replace each changed document atomically
        with measure("store", timer):
            for doc, chunks in prepared:
                source = doc.metadata["source"]
                doc_kept = [c for c in kept if c.metadata.get("source") == source]
                front, body = parse_front_matter(doc.content)
                doc_info = {
                    "ingest_sig": doc.metadata["ingest_sig"],
                    "doc_hash": content_hash(body),
                    "domain": self.domains.domain_for(source, front),
                    "title": front.get("title") or document_title(body, source),
                    "duplicates_dropped": dropped_by_source.get(source, 0),
                }
                self.store.replace_source(source, doc_kept, doc_info)
                results.append({
                    "file": source,
                    "status": "ingested",
                    "chunks_created": len(doc_kept),
                    "duplicates_dropped": doc_info["duplicates_dropped"],
                    "doc_hash": doc_info["doc_hash"],
                    "domain": doc_info["domain"],
                })
        stage_ms["store"] = timer["store"]["ms"]

        # 6. Prune documents that disappeared from disk
        if prune:
            with measure("prune", timer):
                present = {d.metadata["source"] for d in documents}
                if not present:
                    print("  prune skipped: no documents found (refusing to empty the store)")
                else:
                    for source in set(stored) - present:
                        self.store.delete_by_source(source)
                        results.append({"file": source, "status": "pruned", "chunks_created": 0})
            stage_ms["prune"] = timer["prune"]["ms"]

        total = sum(r["chunks_created"] for r in results)
        total_ms = round(sum(stage_ms.values()), 1)
        perf.update({
            "stage_ms": {k: round(v, 1) for k, v in stage_ms.items()},
            "total_ms": total_ms,
            "documents": len(documents),
            "documents_ingested": sum(r["status"] == "ingested" for r in results),
            "documents_unchanged": sum(r["status"] == "unchanged" for r in results),
            "chunks": total,
            "duplicates_dropped": len(self.last_dedup),
            "chunks_per_second": (round(len(all_chunks) / (total_ms / 1000), 1)
                                  if total_ms and all_chunks else None),
            "rss_end_mb": rss_mb(),
            "process_peak_mb": peak_rss_mb(),
        })
        self.last_perf = perf

        print(f"\nIngestion completed: {perf['documents_ingested']} ingested, "
              f"{perf['documents_unchanged']} unchanged, {total} chunks stored, "
              f"{perf['duplicates_dropped']} duplicates dropped")
        print(f"Ingestion perf: {total_ms} ms total ("
              + ", ".join(f"{k} {v}" for k, v in perf["stage_ms"].items())
              + f") | RSS {perf['rss_end_mb']} MB, peak {perf['process_peak_mb']} MB")
        return results

# ---- Usage ----
if __name__ == "__main__":
    pipeline = IngestionPipeline(
        chunk_strategy="recursive",
        chunk_size=512,
        embedding_model="all-MiniLM-L6-v2",
    )

    # Ingest the docs
    pipeline.ingest_directory("./documents")

    # Quick sanity check - search for something
    query = "What is the refund policy?"
    query_vec = pipeline.embedder.embed_text(query)
    results = pipeline.store.search(query_vec, top_k = 3)

    print(f"\nSearch: '{query}'")
    for r in results:
        print(f" [{r['similarity']:.3f}] {r['content'][:100]}...")