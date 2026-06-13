"""
FastAPI application entry point.

This is where everything connects: guardrails -> RAG pipeline -> response.
One command starts the whole system: uvicorn src.api.main:app --reload
"""

import os
import html
import time
import traceback
from contextlib import asynccontextmanager
from fastapi import FastAPI, Security, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from src.api.auth import verify_api_key
from src.api.models import ChatRequest, IngestRequest

load_dotenv()

# Groq key fallback
GROQ_KEYS = [k for k in [os.getenv("GROQ_API_KEY"), os.getenv("GROQ_API_KEY_2"), os.getenv("GROQ_API_KEY_3")] if k]
_key_index = 0

def swap_groq_key():
    global _key_index
    if len(GROQ_KEYS) > 1:
        _key_index = (_key_index + 1) % len(GROQ_KEYS)
        os.environ["GROQ_API_KEY"] = GROQ_KEYS[_key_index]
        print(f"Swapped to Groq key #{_key_index + 1}")

# Shared instances - initialized once at startup
app_state = {}

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add OWASP-recommended security headers to every response."""
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        
        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"

        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"

        # XSS protection (legacy browsers)
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Content Security Policy
        response.headers["Content-Security-Policy"] = "default-src 'self'"

        # Prevent referrer leakage
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # HSTS - force HTTPS (enable when you have TLS)
        # response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        # Prevent caching of sensitive responses
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"

        return response


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
    allow_methods = ["GET", "POST"],
    allow_headers = ["Content-Type", "X-API-Key"],
    allow_credentials=False,
)

app.add_middleware(SecurityHeadersMiddleware)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts = ["localhost", "127.0.0.1"]
)

@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    """Prevent oversized requests that could exhaust memory."""
    MAX_BODY_SIZE = 1_000_000  # 1 MB

    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_BODY_SIZE:
        raise HTTPException(413, "Request body too large")
    
    return await call_next(request)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Never expose internal errors to the client.
    
    Log the full traceback server-side for debugging.
    REutrn a generic message to the user.
    """
    # Log full error server-side
    print(f"ERROR: {request.url.path}")
    traceback.print_exc()

    # Return generic message to client - no stack traces,
    # no databse connection strings, no file paths
    return JSONResponse(
        status_code = 500,
        content = {"error": "An internal error occurred. Please try again later."}
    )

# -- Routes --

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(request: ChatRequest, api_key: str = Security(verify_api_key)):
    pipeline = app_state["pipeline"]
    for attempt in range(3):
        try:
            result = pipeline.answer(query = request.query, top_k = request.top_k)
            break
        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e) or "quota" in str(e).lower():
                swap_groq_key()
                time.sleep(5)
                continue
            raise
    else:
        raise HTTPException(status_code=429, detail="All API keys exhausted")
    result["answer"] = html.escape(result["answer"])  # Sanitize output to prevent XSS
    return result


@app.post("/api/ingest")
async def ingest(request: IngestRequest, api_key: str = Security(verify_api_key)):
    """Ingest documents into the knowledge base."""

    store = app_state["store"]
    stats = store.get_stats()

    # Warn if re-ingesting would affect many existing chunks
    if stats["total_chunks"] > 1000 and not request.confirm:
        return {
            "warning": f"Knowledge base has {stats['total_chunks']} chunks."
                    f"Re-ingestion will replace existing chunks from matching files."
                    f"Send 'confirm': true to proceed.",
            "requires_confirmation": True,
        }

    from  src.ingestion.ingest import IngestionPipeline

    pipeline = IngestionPipeline(
        chunk_strategy = request.chunk_strategy,
        chunk_size = request.chunk_size,
        embedding_model= os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        db_connection = os.getenv("DATABASE_URL"),
    )

    results = pipeline.ingest_directory(request.directory)

    # Rebuild BM25 index after new ingestion
    bm25 = app_state.get("bm25")
    if bm25:
        bm25.load_from_vector_store(app_state["store"])

    return {"status": "ok", "files_ingested": len(results), "details": results}

@app.post("/api/evaluate")
async def evaluate(request: dict, api_key: str = Security(verify_api_key)):
    """Admin endpoint - run RAGAS evaluation."""
    from src.evaluation.eval_dataset import EvalDataset, build_sample_dataset
    from src.evaluation.metrics import RAGMetrics
    from src.evaluation.ragas_eval import EvaluationRunner

    label = request.get("label", "default")
    dataset_path = request.get("dataset", None)

    if dataset_path:
        dataset = EvalDataset().load(dataset_path)
    else:
        dataset = build_sample_dataset()

    pipeline = app_state["pipeline"].rag # unwrap from GuardedRAGPipeline
    metrics = RAGMetrics()
    runner = EvaluationRunner(pipeline, metrics, dataset)
    report = runner.run(config_label = label)
    runner.save_report(report)

    return report