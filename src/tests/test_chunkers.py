"""
Unit tests for chunking strategies.
"""

import pytest
from src.ingestion.document_loader import Document
from src.ingestion.chunkers import FixedSizeChunker, RecursiveChunker
from src.ingestion.chunker_factory import ChunkerFactory


@pytest.fixture
def sample_doc():
    return Document(
        content="""# Refund Policy

        All purchases are eligible for a full refund within 30 days of purchase.
        Items must be returned in original packaging with receipt.

        After 30 dyas, we offer store credit equal to the purchase price.
        Store credit never expires and can be used on any product.

        # Shipping Policy

        Standard shipping takes 5-7 business days. Express shipping is available
        for an additional $15 and delivers within 2 business days.""",
            metadata={"source": "policies.md", "filename": "policies.md"}
    )

class TestFixedSizeChunker:

    def test_creates_chunks(self, sample_doc):
        chunker = FixedSizeChunker(chunk_size=200, overlap=30)
        chunks = chunker.chunk(sample_doc)
        # All chunks except possibly the last should be near chunk_size
        for chunk in chunks[:-1]:
            assert len(chunk.content) <= 250 # some tolerance

    def test_overlap_exists(self, sample_doc):
        chunker = FixedSizeChunker(chunk_size=200, overlap=50)
        chunks = chunker.chunk(sample_doc)
        if len(chunks) >= 2:
            # End of chunk 0 should overlap with start of chunk 1
            end_of_first = chunks[0].content[-50:]
            assert end_of_first in chunks[1].content

    def test_metadata_preserved(self, sample_doc):
        chunker = FixedSizeChunker(chunk_size=200, overlap=30)
        chunks = chunker.chunk(sample_doc)
        for chunk in chunks:
            assert chunk.metadata["source"] == "policies.md"
            assert chunk.metadata["chunk_strategy"] == "fixed_size"

    def test_overlap_must_be_less_than_chunk_size(self):
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=100, overlap=100)

    def test_empty_document(self):
        doc = Document(content="", metadata={"source": "empty.md"})
        chunker = FixedSizeChunker(chunk_size=200, overlap=30)
        chunks = chunker.chunk(doc)
        assert len(chunks) == 0

class TestRecursiveChunker:

    def test_creates_chunks(self, sample_doc):
        chunker = RecursiveChunker(chunk_size=200)
        chunks = chunker.chunk(sample_doc)
        assert len(chunks) > 0

    def test_splits_on_paragraphs(self, sample_doc):
        chunker = RecursiveChunker(chunk_size=300)
        chunks = chunker.chunk(sample_doc)
        # Should not cut mid-sentence when paragraph breaks exist
        for chunk in chunks:
            assert not chunk.content.endswith(" days")

    def test_no_empty_chunks(self, sample_doc):
        chunker = RecursiveChunker(chunk_size=200)
        chunks = chunker.chunk(sample_doc)
        for chunk in chunks:
            assert len(chunk.content.strip()) > 0

    def test_all_content_preserved(self, sample_doc):
        chunker = RecursiveChunker(chunk_size=500)
        chunks = chunker.chunk(sample_doc)
        combined = "".join(chunk.content for chunk in chunks)
        # key phrases should survive chunking
        assert "refund" in combined.lower()
        assert "shipping" in combined.lower()

    def test_long_paragraph_does_not_repeat_text(self):
        """A paragraph over chunk_size must not re-emit or re-glue the chunk before it."""
        short_a = "Intro paragraph about the alpha setting."
        long_b = " ".join(f"Sentence {i} explains option {i} for admins." for i in range(30))
        short_c = "Closing paragraph about the gamma setting."
        doc = Document(content=f"{short_a}\n\n{long_b}\n\n{short_c}", metadata={"source": "t.md"})
        texts = [c.content for c in RecursiveChunker(chunk_size=200).chunk(doc)]
        assert len(texts) == len(set(texts))                         # no duplicate chunks
        assert sum(short_a in t for t in texts) == 1                 # earlier text appears once
        assert sum(short_c in t for t in texts) == 1
        assert all(len(t) <= 200 for t in texts)


class TestChunkerFactory:
    def test_create_fixed(self):
        chunker = ChunkerFactory.create("fixed", chunk_size=200, overlap=30)
        assert isinstance(chunker, FixedSizeChunker)

    def test_create_recursive(self):
        chunker = ChunkerFactory.create("recursive", chunk_size=200)
        assert isinstance(chunker, RecursiveChunker)

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            ChunkerFactory.create("nonexistent")