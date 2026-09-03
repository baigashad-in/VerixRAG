"""Contextual chunks: the section path is embedded and BM25-indexed with each chunk."""

import pytest

from src.config import reset_settings
from src.ingestion.embedding_service import EmbeddedChunk
from src.ingestion.metadata import chunk_context, index_text
from src.retrieval.sparse_retrieval import BM25Retriever
from src.tests import fakes


def test_chunk_context_and_index_text():
    assert chunk_context({"section": "Guide › Password Reset", "title": "Guide"}) == "Guide › Password Reset"
    assert chunk_context({"title": "Guide"}) == "Guide"
    assert chunk_context({}) == ""
    assert index_text("Body.", {"context": "Guide › Reset"}) == "Guide › Reset\n\nBody."
    assert index_text("Body.", {}) == "Body." and index_text("Body.", None) == "Body."


def test_bm25_indexes_the_context_line():
    chunks = [{"id": 1, "content": "The link expires after 24 hours.",
               "metadata": {"context": "Account Guide › Password Reset"}},
              {"id": 2, "content": "Standard delivery takes five days.", "metadata": {}}]
    bm25 = BM25Retriever()
    bm25.index(chunks)
    assert [r.chunk_id for r in bm25.search("password reset", top_k=2)] == [1]
    assert bm25.search("password reset", top_k=2)[0].content == chunks[0]["content"]   # stored text unchanged


def test_embedding_uses_context(fake_models):
    _, _, embedder = fakes.build_engine([("x", {})])
    chunk = EmbeddedChunk(content="The link expires after 24 hours.", embedding=[],
                          metadata={"context": "Guide › Password Reset"})
    [out] = embedder.embed_chunks([chunk])
    assert out.embedding == embedder.embed_text("Guide › Password Reset\n\nThe link expires after 24 hours.")
    assert out.content == chunk.content


LONG_DOC = "# Field Guide\n\n## Zebracode Settings\n\n" + " ".join(
    f"Sentence {i} explains how the dashboard option number {i} behaves for admins." for i in range(40))


@pytest.mark.db
def test_ingest_stores_context_and_bm25_finds_later_chunks(clean_db, fake_models, tmp_path, monkeypatch):
    from src.ingestion.ingest import IngestionPipeline

    kb = tmp_path / "documents"
    kb.mkdir()
    (kb / "guide.md").write_text(LONG_DOC)
    monkeypatch.setenv("DATABASE_URL", clean_db)

    stores = []

    def ingest(contextual: bool):
        monkeypatch.setenv("CONTEXTUAL_CHUNKS", "true" if contextual else "false")
        reset_settings()
        pipe = IngestionPipeline(db_connection=clean_db, documents_root=str(kb))
        stores.append(pipe.store)
        return pipe, pipe.ingest_directory(str(kb))

    try:
        pipe, first = ingest(False)
        plain = BM25Retriever()
        plain.load_from_vector_store(pipe.store)
        plain_hits = len(plain.search("zebracode", top_k=50))

        pipe, second = ingest(True)
        assert second[0]["status"] == "ingested"          # toggling re-ingests (new signature)
        with pipe.store.conn.cursor() as cur:
            cur.execute("SELECT metadata->>'context', metadata->>'section' FROM chunks")
            rows = cur.fetchall()
        pipe.store.conn.commit()
        assert len(rows) > 1 and all(ctx == sec for ctx, sec in rows)
        ctx = BM25Retriever()
        ctx.load_from_vector_store(pipe.store)
        assert len(ctx.search("zebracode", top_k=50)) == len(rows) > plain_hits

        _, third = ingest(True)
        assert third[0]["status"] == "unchanged"
    finally:
        for store in stores:                  # reads leave transactions open; release them
            store.close()

@pytest.mark.parametrize("contextual", [True, False])
def test_prepare_sets_context_only_when_enabled(contextual):
    """No database: catches a missing or misplaced context line in IngestionPipeline._prepare."""
    from src.ingestion.chunker_factory import ChunkerFactory
    from src.ingestion.document_loader import Document
    from src.ingestion.ingest import IngestionPipeline
    from src.retrieval.scoping import load_domain_config

    pipe = IngestionPipeline.__new__(IngestionPipeline)
    pipe.chunker = ChunkerFactory.create(strategy="recursive", chunk_size=512)
    pipe.domains = load_domain_config("config/domains.json")
    pipe.contextual = contextual
    doc = Document(content=LONG_DOC, metadata={"source": "guide.md", "ingest_sig": "x"})
    chunks = pipe._prepare(doc)
    assert len(chunks) > 1
    for c in chunks:
        if contextual:
            assert c.metadata["context"] == c.metadata["section"]
        else:
            assert "context" not in c.metadata
    if contextual:
        assert any(c.metadata["context"] == "Field Guide › Zebracode Settings" for c in chunks)