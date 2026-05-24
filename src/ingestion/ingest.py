"""
The complete ingestion pipeline.
Document -> Chunks -> Embeddings -> Vector Store

This script is run when new documents are added to the knowledge base. 
"""

from src.ingestion.document_loader import DocumentLoader, Document
from src.ingestion.chunkers import FixedSizeChunker, RecursiveChunker, SemanticChunker
from src.ingestion.embedding_service import EmbeddingService
from src.storage.vector_store import VectorStore
from src.ingestion.chunker_factory import ChunkerFactory

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
                 db_connection: str = "postgresql://user:pass@localhost:5432/ragdb"):
        
        self.loader = DocumentLoader()
        self.chunker = ChunkerFactory.create(
            strategy=chunk_strategy,
            chunk_size=chunk_size,
        )
        self.embedder = EmbeddingService(model_name=embedding_model)
        self.store = VectorStore(
            connection_string=db_connection,
            embedding_dimension=self.embedder.dimension,
        )

        # Log the configuration - critical for experiment tracking
        self.config = {
            "chunk_strategy": chunk_strategy,
            "chunk_size": chunk_size,
            "embedding_model": embedding_model,
        }
        print(f"Pipeline configured: {self.config}")

    def ingest_file(self, file_path: str) -> dict:
        """Ingest a single file end-to-end."""
        # 1. Load the document
        document = self.loader.load_file(file_path)
        print(f"Loaded: {file_path} ({document.metadata['char_count']} chars)")

        # 2. Chunk
        chunks = self.chunker.chunk(document)
        print(f"Chunked into {len(chunks)} pieces")

        # 3. Embed
        embedded = self.embedder.embed_chunks(chunks)

        # 4. Store
        self.store.delete_by_source(file_path) # remove stale data
        self.store.insert_chunks(embedded)

        return {
            "file": file_path,
            "chunks_created": len(chunks),
            "config": self.config,
        }
    
    def ingest_directory(self, dir_path: str) -> list[dict]:
        """Ingest all documents in a directory."""
        documents = self.loader.load_directory(dir_path)
        results = []

        for doc in documents:
            source = doc.metadata["source"]
            chunks = self.chunker.chunk(doc)
            embedded = self.embedder.embed_chunks(chunks)
            self.store.delete_by_source(source)
            self.store.insert_chunks(embedded)

            results.append({
                "file": source,
                "chunks_created": len(chunks),
            })
        
        total = sum(r["chunks_created"] for r in results)
        print(f"\nIngestion completed: {len(results)} files, "
              f"{total} total chunks")
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
        print(f" [{r['similarity']}:.3f] {r['content'][:100]}...")
