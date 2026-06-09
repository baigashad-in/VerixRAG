"""
BM25 sparse retrieval - keyword-based search.

HOW BM25 WORKS:
1. Tokenize query into individual words
2. For each document, score = sum of term scores
3. Term score considers:
    - TF (term frequency): how often does the word apprear in this doc?
    - IDF (inverse document frequency): how rare is the word across all docs? Rare words matter more. "the" scores low. "E-4012" scores high.
    - Document length normalization : long docs don't get unfair advantage

WHY NOT JUST USE 'word in text'?
Because "the refund policy" would match every document containing "the." BM25's IDF scoring means common words contribute almost nothing, while specific terms drive the ranking.
"""

from dataclasses import dataclass
from rank_bm25 import BM25Okapi
import re
from src.storage.vector_store import VectorStore

@dataclass
class SparseResult:
    """A search result from BM25 with its relevance score."""
    chunk_id: int
    content: str
    metadata: dict
    score: float

class BM25Retriever:
    """Keyword-based retrieval using BM25.

    This lives alongside the vector store - it indexes the same
    chunks but searches them differently. Think of it as a
    second lens on the same data.
    """

    def __init__(self):
        self.documents = [] # raw chunk data
        self.bm25 = None    # the BM25 index
        self._tokenized = [] # tokenized version of docs

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization: lowercase, split, remove noise.

        In production, we might add:
        - Stemming (running -> run)
        - Stopword removal (the, is, at)
        - Domain-specific tokens
        
        """
        text = text.lower()
        # Keep alphanumeric and hyphens (for codes like E-4012)
        tokens = re.findall(r'[a-z0-9][\w-]*', text)
        return tokens

    def index(self, chunks: list[dict]):
        """Build the BM25 index from chunk data.
        
        Args:
            chunks: list of dicts with 'id', 'content', 'metadata'
            (same format returned by VectoreStore.search or
            a direct DB query)
        Call this once after ingestion, or rebuild when docs change.
        """
        if not chunks:
            self.bm25 = None
            self._chunks = []
            self._tokenized = []
            return

        self.documents = chunks
        self._tokenized = [
            self._tokenize(chunk["content"])
            for chunk in chunks
        ]

        #BM25Okapi builds its term frequency / IDF statistics here
        self.bm25 = BM25Okapi(self._tokenized)
        print(f"BM25 index built: {len(chunks)} documents")

    def search(self, query: str, top_k: int = 5) -> list[SparseResult]:
        """Search using BM25 keyword matching."""
        if self.bm25 is None:
            return []
        
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indicies sorted by score (descending)
        top_indices = sorted(
            range(len(scores)),
            key = lambda i:scores[i],
            reverse = True
        )[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0: # skip zero-score matches
                results.append(SparseResult(
                    chunk_id = self.documents[idx].get("id", idx),
                    content = self.documents[idx]["content"],
                    metadata = self.documents[idx].get("metadata", {}),
                    score = float(scores[idx]),
                ))

        return results
    

    def load_from_vector_store(self, vector_store) -> int:
        """Load all chunks from the vecttor store into BM25.
        
        This keeps boath indexes in sync - they search the same
        data with different methods.
        """

        with vector_store.conn.cursor() as cur:
            cur.execute(
                "SELECT id, content, metadata FROM chunks"
            )
            chunks = [
                {"id": row[0], "content": row[1], "metadata": row[2]}
                for row in cur.fetchall()
            ]

        self.index(chunks)
        return len(chunks)


    