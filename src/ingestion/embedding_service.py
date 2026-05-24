"""
Embedding service that converts text into dense vectors.

MODEL CHOICE:
- OpenAI text-embedding-3-small: easy, costs money, vendor lock-in
- BGE-M3 (BAAI): free, multilingual, runs locally, very competitive
- all-MiniLM-L6-v2: lightweight, fast, good for prototyping

"""

import numpy as np

from dataclasses import dataclass
from sentence_transformers import SentenceTransformer


@dataclass
class EmbeddedChunk:
    """A chunk with its vector representation attached.
    Gets stored in the database - the text for
    display, the vector for search, the metadata for filtering.
    """
    content: str
    embedding: list[float]
    metadata: dict


class EmbeddingService:
    """ Generates embeddings using sentence-transformers.

    WHY not just call OPENAI?
    1. Cost - embedding 100k chunks with OpenAI adds up fast
    2. Latency - local models have no network round-trip
    3. Privacy - your documents never leave your machine
    
    In production we use OpenAI for quality and cache 
    aggressively.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_embedding_dimension()
        print(f"Loaded embedding model: {model_name} " 
              f"(dimension: {self.dimension})")
        
    def embed_text(self, text: str) -> list[float]:
        """Embed a single piece of text. Used for queries."""
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.tolist()
    
    def embed_chunks(self, chunks: list) -> list[EmbeddedChunk]:
        """Embed many chunks efficiently in a single batch.
        
        Why batch?
        Embedding one-at-a-time is ~10x slower than batching.
        The model can process multiple texts on GPU in parallel.
        For 10,000 chunks, this is the difference between
        2 minutes and 20 minutes.
        """

        texts = [chunk.content for chunk in chunks]

        # batch encode - this is the performance-critical call
        vectors = self.model.encode(
            texts,
            normalize_embeddings = True, # needed for cosine similarity
            show_progress_bar = True,
            batch_size = 64, # tune based on GPU memory
        )

        embedded = []
        for chunk, vector in zip(chunks, vectors):
            embedded.append(EmbeddedChunk(
                content = chunk.content,
                embedding = vector.tolist(),
                metadata = chunk.metadata,
            ))

        print(f"Embedded {len(embedded)} chunks "
              f"({self.dimension}-dim vectors)")
        return embedded
    
