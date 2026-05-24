"""Vector store using PostgreSQL + pgvector.

Why pgvector over Pinecone, Weaviate, or FAISS?
Because it provides hybrid search with metadata
filters in a single query, without adding another
managed service.

SCHEMA DESIGN:
Each row = one chunk, with its text, vector, and metadata
as JSONB. The JSONB column lets you filter by source file,
chunk strategy, date, or any custom field without schema
migrations.
"""

import json
import re
import psycopg2
from psycopg2.extras import execute_values

class VectorStore:
    ALLOWED_KEYS = {
            "filename", "source", "file_type", 
            "chunk_strategy", "chunk_index",
    }

    # Content size limit per chunk — prevents storing huge blobs
    MAX_CHUNK_CONTENT = 100_000 # 100k chars - adjust based on your needs and embedding model limits

    def __init__(self, connection_string: str, embedding_dimension: int =384):
        """
        Args:
            connection_string: PostgreSQL connection string e.g. "postgresql://user:pass@localhost:5432/ragdb"
            embedding_dimension: must match the embedding model
            all-MiniLM-L6-v2 = 384
            BGE-M3 = 1024
            OpenAI text-embedding-3-small = 1536
        """
        self.conn = psycopg2.connect(connection_string)
        self.dimension = embedding_dimension
        self._initialize_db()

    def _initialize_db(self):
        """Create the pgvector extension and chunks table.
        The HNSW INDEX is the key performance piece:
        Without it, every search scans ALL vectors (exact search).
        With HNSW, it searched an approximate graph structure - 
        ~100x faster at the cost of occasionally missing the
        absolute best match (99.5%+ recall in practice).

        m = 16: connections per node in the graph (higher = more
        accurate but more memory and slower inserts(more connections to build))
        ef_construction = 64: build-time thoroughness (higher = better index but slower to build)
        """
        with self.conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            cur.execute(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                embedding vector({self.dimension}) NOT NULL,
                metadata JSONB DEFAULT '{{}}'::jsonb,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # HNSW index for fast approximate nearest neighbor search
            cur.execute(f"""
            CREATE INDEX IF NOT EXISTS chunks_embeddings_idx
                        ON chunks
                        USING hnsw (embedding vector_cosine_ops)
                        WITH (m = 16, ef_construction = 64);
            """)

            # Index on metadata for filtered queries
            cur.execute("""
            CREATE INDEX IF NOT EXISTS chunks_metadata_idx
                        ON chunks
                        USING gin (metadata);
            """)
            self.conn.commit()
        print("Vector Store initialized")
    
    def insert_chunks(self, embedded_chunks:list) -> int:
        """Bulk insert embedded chunks into the store.
        Uses execute_values for batch insertion - much faster
            than individual INSERTS for large document sets.
        """
        with self.conn.cursor() as cur:
            data = [
                (
                    chunk.content,
                    chunk.embedding, # pgvector handles list -> vector
                    json.dumps(chunk.metadata)
                )
                for chunk in embedded_chunks
            ]

            execute_values(
                cur,
                """INSERT INTO chunks (content, embedding, metadata) VALUES %s""",
                data,
                template = "(%s, %s::vector, %s::jsonb)",
            )
            self.conn.commit()

        print(f"Inserted {len(embedded_chunks)} chunks")
        return len(embedded_chunks)
    
    def search(self, query_embedding: list[float], top_k: int = 5,
                metadata_filter: dict = None) -> list[dict]:
        """Find the most similar chunks to a query vector.
        This is the core of RAG retrieval. 
        
        The <=> operator computes cosine distance (1 - similarity).
        Lower distance = more similar. We ORDER BY distance ASC to
        get the best matches first.

        METADATA FILTERING is what makes this production-grade:
        you can search ONLY within a specific document, or only chunks from a certain strategy, or only recent documents.

        Example: search only in the refund policy doc:
            metadata_filter = {"filename": "refund_policy.md"}
        """

        with self.conn.cursor() as cur:
            # Build the query with optional metadata filter
            where_clause = ""
            params = [query_embedding, top_k]

            if metadata_filter:

                conditions = []
                for key, value in metadata_filter.items():
                    
                    if key not in self.ALLOWED_KEYS:
                        raise ValueError(f"Invalid metadata key: {key}")
                    conditions.append(f"metadata->>'{key}' = %s"
                                        )
                    params.insert(-1, value) # before top_k
                where_clause = "WHERE " + " AND ".join(conditions)

            cur.execute(f"""
            SELECT
                id,
                content,
                metadata,
                1 - (embedding <=> %s::vector) AS similarity
                FROM chunks
                {where_clause}
                ORDER BY embedding <=> %s::vector 
                LIMIT %s;""",
                [query_embedding] + params[1:-1] +
                [query_embedding, top_k])
            
            results = []
            for row in cur.fetchall():
                results.append({
                    "id": row[0],
                    "content": row[1],
                    "metadata": row[2],
                    "similarity": float(row[3])
                })
            return results
    def get_stats(self) -> dict:
        """Get store statistics - useful for debugging."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunks;")
            count = cur.fetchone()[0]

            cur.execute("""SELECT metadata->>'chunk_strategy', COUNT(*) FROM chunks GROUP BY metadata->>'chunk_strategy';""")
            by_strategy = dict(cur.fetchall())
        return {"total_chunks": count, "by_strategy": by_strategy}
    
    def delete_by_source(self, source: str):
        """Delete all chunks from a specific source file.
        
        Essential for re-ingestion when a document is updated,
        you delete old chunks and re-ingest. Without this, you'd
        have duplicate/stale chunks polluting your results.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                "DELETE FROM chunks WHERE metadata->>'source' = %s", (source,)
            )
            deleted = cur.rowcount
            self.conn.commit()
        print(f"Deleted {deleted} chunks from {source}")
        return deleted
        
