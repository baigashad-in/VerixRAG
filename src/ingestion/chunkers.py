import re
import numpy as np

from dataclasses import dataclass

@dataclass
class Chunk:
    """A piece of a document, ready to be embedded.
    
    Each chunk carries its parent document's metadata PLUS its own position info. This is how we trace back to sources.
    """

    content: str
    metadata: dict # inherited from document + chunk-specific info
    chunk_index: int

class FixedSizeChunker:
    """Split text into fixed-size character windows with overlap.
    
    Overlap ensures boundary information appears in multiple chunks, improving context for embeddings.

    Typical starting point: chunk_size=512, overlap=50
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        if overlap >= chunk_size:
            raise ValueError("Overlap must be smaller than chunk size.")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document) -> list[Chunk]:
        text = document.content
        chunks = []
        start = 0
        index = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            # Don't create tiny trailing chunks - they're noise
            if len(chunk_text.strip()) < 50:
                # Append to previous chunk instead
                if chunks:
                    prev = chunks[-1]
                    chunks[-1] = Chunk(
                        content = prev.content + chunk_text,
                        metadata = prev.metadata,
                        chunk_index = prev.chunk_index,
                    )
                break

            chunks.append(Chunk(
                content = chunk_text,
                metadata = {
                    **document.metadata,
                    "chunk_index": index,
                    "chunk_strategy": "fixed_size",
                    "chunk_size": self.chunk_size,
                    "char_start": start,
                    "char_end": min(end, len(text)),
                },
                chunk_index = index,
            ))

            start += self.chunk_size - self.overlap
            index += 1

        return chunks

class RecursiveChunker:
    """Split on natural text boundaries, falling back to smaller ones.

    The idea: Try to split on paragraph breaks first (\n\n).
    If a paragraph is still too big, split on sentence breaks (. ).
    If a sentence is still too big, split on words ( ).

    This preserves meaning much better than fixed-size, because
    paragraphs are natural units of thought.

    This is the strategy LangChain's RecursiveCharacterTextSplitter uses. We are building it from scratch.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        # Try these separators in order: most meaningful -> least
        self.separators = ["\n\n", "\n", ". ", " "]

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using progressively finer separators."""

        if len(text) <= self.chunk_size:
            return [text]

        # Find the best separator that actually exists in the text
        separator = separators[0] if separators else " "
        remaining_seps = separators[1:] if len(separators) > 1 else []

        parts = text.split(separator)

        chunks = []
        current_chunk = ""

        for part in parts:
            # Would adding this part exceed our limit?
            candidate = (current_chunk + separator + part
                         if current_chunk else part)
            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                # Save current chunk if we have one
                if current_chunk:
                    chunks.append(current_chunk)

                # If this single part is too big, recurse with
                # finer separators
                if len(part) > self.chunk_size and remaining_seps:
                    sub_chunks = self._split_text(part, remaining_seps)
                    chunks.extend(sub_chunks)
                else:
                    current_chunk = part

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def chunk(self, document) -> list[Chunk]:
        raw_chunks = self._split_text(document.content, self.separators)

        return [
            Chunk(
                content = text.strip(),
                metadata = {
                    **document.metadata,
                    "chunk_index": i,
                    "chunk_strategy": "recursive",
                },
                chunk_index = i,
            )
            for i, text in enumerate(raw_chunks)
            if text.strip() # skip empty chunks
        ]
    
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')  # compile once

class SemanticChunker:
    """Split text where the meaning changes, not at arbitrary positions.
    
    Working:
    1. Split into sentences
    2. Embed each sentence (convert to vector)
    3. Compare adjacent sentence embeddings
    4. Where similarity drops sharply -> that's a topic boundary -> split there

    A document might discuss "pricing" for 3 paragraphs,
    then switch to "installation". Fixed-size chunking might split right in the middle of the pricing section. Semantic chunking keeps all of
    "pricing" together because the embeddings are similar.

    Tradeoff: It's slower (requires embedding every sentence) and needs a good sentence embedding model.
    """

    def __init__(self, embedding_model = None,
                 similarity_threshold: float = 0.5):
        self.embedding_model = embedding_model
        # We'll set this up when we build the embedding layer
        self.similarity_threshold = similarity_threshold

    MAX_INPUT_LENGTH = 5_000_000  # 5M chars max

    def _split_into_sentences(self, text: str) -> list[str]:
        """Simple sentence splitting.
        In production, we use spaCy or nltk for better accuracy."""

        if len(text) > self.MAX_INPUT_LENGTH:
            raise ValueError(
                f"Text too long for semantic chunking: "
                f"{len(text)} chars (max {self.MAX_INPUT_LENGTH})"
            )
        
        sentences = _SENTENCE_SPLIT.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def _cosine_similarity(self, vec_a, vec_b) -> float:
        """How similar are two vectors? 1.0 = identical, 0.0 = unrelated."""
        dot = np.dot(vec_a, vec_b)
        norm = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        return dot / norm if norm > 0 else 0.0
    
    def chunk(self, document) -> list[Chunk]:
        sentences = self._split_into_sentences(document.content)

        if not sentences:
            return []
        
        # Embed all sentences
        embeddings = self.embedding_model.encode(sentences)

        # Find topic boundaries: where do adjacent sentences
        # suddenly become less similar?
        chunks = []
        current_sentences = [sentences[0]]

        for i in range(1, len(sentences)):
            similarity = self._cosine_similarity(
                embeddings[i-1], embeddings[i]
            )

            if similarity < self.similarity_threshold:
                # Big meaning shift detected - start a new chunk
                chunks.append(" ".join(current_sentences))
                current_sentences = [sentences[i]]
            else:
                # Still on the same topic - keep accumulating
                current_sentences.append(sentences[i])
        
        # Add the last group
        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return [
            Chunk(
                content = text,
                metadata = {
                    **document.metadata,
                    "chunk_index": i,
                    "chunk_strategy": "semantic",
                },
                chunk_index = i,
            )
            for i, text in enumerate(chunks)
        ]