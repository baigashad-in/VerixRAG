"""Vector store using PostgreSQL + pgvector.

It provides hybrid search with metadata filters in a 
single query, without adding another managed service.

SCHEMA DESIGN:
Each row = one chunk, with its text, vector, and metadata
as JSONB. The JSONB column filter by source file,
chunk strategy, date, or any custom field without schema
migrations.
"""

import json
import re
import time
import psycopg
import threading

class VectorStore:
    ALLOWED_KEYS = {
            "filename", "source", "file_type", 
            "chunk_strategy", "chunk_index",
            "doc_hash", "section", "title", "domain",
    }

    # Content size limit per chunk — prevents storing huge blobs
    MAX_CHUNK_CONTENT = 100_000 # 100k chars - adjust based on needs and embedding model limits

    def __init__(self, connection_string: str, embedding_dimension: int =384):
        """
        Args:
            connection_string: PostgreSQL connection string e.g. "postgresql://user:pass@localhost:5432/ragdb"
            embedding_dimension: must match the embedding model
            all-MiniLM-L6-v2 = 384
            BGE-M3 = 1024
            OpenAI text-embedding-3-small = 1536
        """
        self._dsn = connection_string
        self._local = threading.local()          # one connection per thread
        self._connections: list = []
        self._connections_lock = threading.Lock()
        self.dimension = embedding_dimension
        self._version_cache: tuple[float, str] | None = None
        self._initialize_db()

    @property
    def conn(self):
        """Return this thread's connection, opening it on first use."""
        conn = getattr(self._local, "conn", None)
        if conn is None or conn.closed:
            conn = psycopg.connect(self._dsn)
            self._local.conn = conn
            with self._connections_lock:
                self._connections.append(conn)
        return conn
    
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
        # Parameterize dimension safely
        if not isinstance(self.dimension, int) or not (1 <= self.dimension <= 4096):
            raise ValueError(f"Invalid embedding dimension: {self.dimension}")

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

            # Document registry: one row per ingested file and version.
            # Lets ingestion skip unchanged files even when every chunk
            # of a file was dropped as a duplicate.
            cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                source TEXT PRIMARY KEY,
                ingest_sig TEXT,
                doc_hash TEXT,
                domain TEXT,
                title TEXT,
                chunks INTEGER,
                duplicates_dropped INTEGER,
                ingested_at TIMESTAMPTZ DEFAULT now()
            );
            """)
            self.conn.commit()
        print("Vector Store initialized")

    def _validate_metadata(self, metadata: dict) -> str:
        """Sanitize metadata before storing."""
        # Remove any keys that could be problematic
        clean = {}
        for key, value in metadata.items():
            if not re.match(r'^[a-z_][a-z0-9_]*$', key):
                continue # silently skip invalid keys
            # Values: convert to string, limit length
            clean[key] = str(value)[:1000]
        return json.dumps(clean)
    
    def insert_chunks(self, embedded_chunks:list) -> int:
        """Bulk insert embedded chunks into the store.
        Inserts individually within a single transaction —
        the commit at the end makes it efficient.
        """
        with self.conn.cursor() as cur:
            for chunk in embedded_chunks:
                # Validate content size
                if len(chunk.content) > self.MAX_CHUNK_CONTENT:
                    raise ValueError(f"Chunk content exceeds {self.MAX_CHUNK_CONTENT} chars")
                cur.execute(
                    "INSERT INTO chunks (content, embedding, metadata) VALUES (%s, %s::vector, %s::jsonb)",
                    (chunk.content, str(chunk.embedding), self._validate_metadata(chunk.metadata))
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

        METADATA FILTERING:
        search ONLY within a specific document, or only chunks from a certain strategy, or only recent documents.

        Example: search only in the refund policy doc:
            metadata_filter = {"filename": "refund_policy.md"}
        """
        # Validate top_k to prevent absurd queries
        if not isinstance(top_k, int) or not (1 <= top_k <= 100):
            raise ValueError("top_k must be between 1 and 100")

        with self.conn.cursor() as cur:
            # Build the query with optional metadata filter
            where_clause = ""
            params = [str(query_embedding), top_k]

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
                [str(query_embedding)] + params[1:-1] +
                [str(query_embedding), top_k])
            
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
    
    def replace_source(self, source: str, embedded_chunks: list,
                       doc_info: dict | None = None) -> tuple[int, int]:
        """Delete a source's chunks and insert new ones in ONE transaction.

        Readers never see the document missing halfway through a
        re-ingest, and a failed insert leaves the old version in place.
        doc_info (ingest_sig, doc_hash, domain, title, duplicates_dropped)
        is upserted into the documents registry in the same transaction.
        Returns (deleted, inserted).
        """
        if not re.match(r'^[\w\-. ]+$', source):
            raise ValueError(f"Invalid source filename: {source}")
        with self.conn.cursor() as cur:
            try:
                cur.execute("DELETE FROM chunks WHERE metadata->>'source' = %s", (source,))
                deleted = cur.rowcount
                for chunk in embedded_chunks:
                    if len(chunk.content) > self.MAX_CHUNK_CONTENT:
                        raise ValueError(f"Chunk content exceeds {self.MAX_CHUNK_CONTENT} chars")
                    cur.execute(
                        "INSERT INTO chunks (content, embedding, metadata) VALUES (%s, %s::vector, %s::jsonb)",
                        (chunk.content, str(chunk.embedding), self._validate_metadata(chunk.metadata))
                    )
                if doc_info is not None:
                    cur.execute("""
                        INSERT INTO documents (source, ingest_sig, doc_hash, domain, title,
                                               chunks, duplicates_dropped, ingested_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, now())
                        ON CONFLICT (source) DO UPDATE SET
                            ingest_sig = EXCLUDED.ingest_sig, doc_hash = EXCLUDED.doc_hash,
                            domain = EXCLUDED.domain, title = EXCLUDED.title,
                            chunks = EXCLUDED.chunks,
                            duplicates_dropped = EXCLUDED.duplicates_dropped,
                            ingested_at = now();""",
                        (source, doc_info.get("ingest_sig"), doc_info.get("doc_hash"),
                         doc_info.get("domain"), doc_info.get("title"),
                         len(embedded_chunks), doc_info.get("duplicates_dropped", 0)))
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
        self.invalidate_version()
        print(f"Replaced {source}: -{deleted} +{len(embedded_chunks)} chunks")
        return deleted, len(embedded_chunks)

    def index_version(self, max_age_seconds: float = 2.0) -> str:
        """A string that changes whenever chunks are inserted or deleted.

        Cache keys include it, so re-ingesting makes old cache entries
        unreachable. Memoised for a couple of seconds so a burst of
        requests costs one tiny query.
        """
        now = time.monotonic()
        if self._version_cache and now - self._version_cache[0] < max_age_seconds:
            return self._version_cache[1]
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*), COALESCE(MAX(id), 0) FROM chunks;")
            count, max_id = cur.fetchone()
        version = f"{count}:{max_id}"
        self._version_cache = (now, version)
        return version

    def invalidate_version(self):
        self._version_cache = None

    def domain_centroids(self) -> dict[str, list[float]]:
        """Mean embedding of each domain's chunks (pgvector AVG aggregate)."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT metadata->>'domain', AVG(embedding)::text
                FROM chunks
                WHERE metadata ? 'domain' AND metadata->>'domain' <> ''
                GROUP BY metadata->>'domain';""")
            return {row[0]: json.loads(row[1]) for row in cur.fetchall()}

    def source_signatures(self) -> dict[str, str | None]:
        """{source: ingest signature} for every stored document.

        Used to skip re-embedding documents that haven't changed.
        Chunks ingested before signatures existed map to None.
        """
        with self.conn.cursor() as cur:
            # Chunks stored before the registry existed: unknown signature
            cur.execute("SELECT DISTINCT metadata->>'source' FROM chunks;")
            signatures = {row[0]: None for row in cur.fetchall() if row[0]}
            cur.execute("SELECT source, ingest_sig FROM documents;")
            signatures.update({row[0]: row[1] for row in cur.fetchall()})
            return signatures

    def list_documents(self) -> list[dict]:
        """The document registry: version, domain and chunk counts per file."""
        with self.conn.cursor() as cur:
            cur.execute("""SELECT source, doc_hash, domain, title, chunks,
                                  duplicates_dropped, ingested_at
                           FROM documents ORDER BY source;""")
            cols = ["source", "doc_hash", "domain", "title", "chunks",
                    "duplicates_dropped", "ingested_at"]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def nearest_chunks(self, query_embedding: list[float], k: int = 3,
                       exclude_sources: list[str] | None = None) -> list[dict]:
        """Nearest stored chunks, optionally skipping some sources.

        Used by deduplication to compare new chunks against what is
        already stored from OTHER documents.
        """
        exclude = list(exclude_sources or [])
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT id, content, metadata->>'source',
                       1 - (embedding <=> %s::vector) AS similarity
                FROM chunks
                WHERE NOT (COALESCE(metadata->>'source', '') = ANY(%s))
                ORDER BY embedding <=> %s::vector
                LIMIT %s;""",
                (str(query_embedding), exclude, str(query_embedding), k))
            return [{"id": r[0], "content": r[1], "source": r[2], "similarity": float(r[3])}
                    for r in cur.fetchall()]

    def delete_by_source(self, source: str):
        """Delete all chunks from a specific source file.
        
        Essential for re-ingestion when a document is updated,
        delete old chunks and re-ingest. Without this, we would
        have duplicate/stale chunks polluting results.
        """
        # Prevent injection via source — only allow filename patterns
        if not re.match(r'^[\w\-. ]+$', source):
            raise ValueError(f"Invalid source filename: {source}")
        
        with self.conn.cursor() as cur:
            cur.execute(
                "DELETE FROM chunks WHERE metadata->>'source' = %s", (source,)
            )
            deleted = cur.rowcount
            cur.execute("DELETE FROM documents WHERE source = %s", (source,))
            self.conn.commit()
        self.invalidate_version()
        print(f"Deleted {deleted} chunks from {source}")
        return deleted
        
    def close(self):
        """Close every connection opened by any thread."""
        with self._connections_lock:
            for conn in self._connections:
                if not conn.closed:
                    conn.close()
            self._connections.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()