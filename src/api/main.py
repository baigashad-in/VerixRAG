"""
FastAPI application entry point.

This is where everything connects: guardrails -> RAG pipeline -> reponse.
One command starts the whole system: uvicorn src.api.main:app --reload
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

# Shared instances - initialized once at startup
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all components at startup, clean up at shutdown.
    
    WHY LIFESPAN?
    Loading embedding models takes ~2 seconds. Re-ranking models
    take another ~2 seconds. You don't want to load these on
    every request. Lifespan loads them once when the server starts
    and shares them across all requests.
    """

    from src.ingestion.embedding_service import EmbeddingService
    from src.storage.vector_store import VectorStore
    from src.retrieval.sparse_retrieval import BM25Retriever
    from src.retrieval.hybrid_retrieval import HybridRetriever
    from src.retrieval.reranker import CrossEncoderReranker
    from src.retrieval.retrieval_engine import RetrievalEngine
    from src.query.query_transformer import QueryTransformer
    from src.query.enhanced_retrieval import EnhancedRetriever
    from src.generation.generator import RAGGenerator
    from src.generation.hallucination_check import HallucinationDetector
    from src.pipeline import RAGPipeline
    from src.guardrails.guardrails_pipeline import GuardedRAGPipeline

    print("Starting up - loading models...")

    # Core services
    embedder = EmbeddingService(
        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    store = VectorStore(
        connection_string = os.getenv("DATABASE_URL"),
        embedding_dimension = embedder.dimension,
    )

    # Retrieval
    bm25 = BM25Retriever()
    bm25.load_from_vector_store(store)

    reranker = CrossEncoderReranker(
        model_name = os.getenv("RERANKER_MODEL",
                               "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )

    retrieval_engine = RetrievalEngine(
        vector_store = store,
        bm25_retriever = bm25,
        embedding_service = embedder,
        reranker = reranker,
    )

    # Query processing
    transformer = QueryTransformer()
    enhanced = EnhancedRetriever(
        retrieval_engine = retrieval_engine,
        query_transformer = transformer,
        embedding_service = embedder,
    )

    # Generation
    generator = RAGGenerator()
    detector = HallucinationDetector()

    # Full pipeline with guardrails
    rag_pipeline = RAGPipeline(enhanced, generator, detector)
    guarded_pipeline = GuardedRAGPipeline(rag_pipeline)

    app_state["pipeline"] = guarded_pipeline
    app_state["store"] = store
    app_state["embedder"] = embedder

    print("Startup complete - ready to serve")

    yield   # app runs here

    # Shutdown
    store.close()
    print("Shutdown complete")

app = FastAPI(
    title = "VerixRAG",
    description = "Production-grade RAG pipeline with evaluation",
    version = "1.0.0",
    lifespan = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["http://localhost:3000"], # React frontend
    allow_methods = ["*"],
    allow_headers = ["*"],
)

# -- Routes --

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(request: dict):
    query = request.get("query", "").strip()

    if not query:
        return {"error": "Query is required."}
    if len(query) > 2000:
        return {"error": "Query too long (max 2000 chars)"}
    
    pipeline = app_state["pipeline"]
    result = pipeline.answer(query = query)

    return result

@app.post("/api/ingest")
async def ingest(request: dict):
    """Ingest documents into the knowledge base."""
    directory = request.get("directory", "./documents")
    strategy = request.get("chunk_strategy", "recursive")
    chunk_size = request.get("chunk_size", 512)

    from  src.ingestion.ingest import IngestionPipeline

    pipeline = IngestionPipeline(
        chunk_strategy = strategy,
        chunk_size = chunk_size,
        embedding_model= os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        db_connection = os.getenv("DATABASE_URL"),
    )

    results = pipeline.ingest_directory(directory)

    # Rebuild BM25 index after new ingestion
    bm25 = app_state.get("bm25")
    if bm25:
        bm25.load_from_vector_store(app_state["store"])

    return {"status": "ok", "files_ingested": len(results), "details": results}