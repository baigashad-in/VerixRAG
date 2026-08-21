"""
FastAPI application entry point.

Everything connects here: guardrails -> RAG pipeline -> response.
One command that starts the whole system: 
        uvicorn src.api.main:app --reload
"""

import os
import html
import time
import traceback
from contextlib import asynccontextmanager
from fastapi import FastAPI, Security, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from dotenv import load_dotenv

from src.ingestion.ingest import IngestionPipeline
from src.ingestion.embedding_service import EmbeddingService
from src.storage.vector_store import VectorStore
from src.retrieval.sparse_retrieval import BM25Retriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retrieval_engine import RetrievalEngine
from src.retrieval.scoping import DomainScorer
from src.query.query_transformer import QueryTransformer
from src.query.enhanced_retrieval import EnhancedRetriever
from src.generation.generator import RAGGenerator
from src.generation.hallucination_check import HallucinationDetector
from src.pipeline import RAGPipeline
from src.guardrails.guardrails_pipeline import GuardedRAGPipeline
from src.guardrails.query_classifier import (
        LLMClassifier, QueryClassifierPipeline, keyword_classifier_from_settings,
    )
from src.evaluation.eval_dataset import EvalDataset, build_sample_dataset
from src.evaluation.metrics import RAGMetrics
from src.evaluation.ragas_eval import EvaluationRunner
from src.cache.response_cache import ResponseCache
from src.config import get_settings

from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from src.api.auth import verify_api_key
from src.api.models import ChatRequest, IngestRequest

from src.observability.tracing import (
    get_trace_store, measure, peak_rss_mb, rss_mb, start_trace, span,
)

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
    take another ~2 seconds. Don't need to load these on
    every request. Lifespan loads them once when the server starts
    and shares them across all requests.
    """

    print("Starting up - loading models...")
    startup_perf = {"rss_at_start_mb": rss_mb()}

    # Core services
    with measure("load_embedder", startup_perf):
        embedder = EmbeddingService(
            model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        )
    with measure("connect_vector_store", startup_perf):
        store = VectorStore(
            connection_string = os.getenv("DATABASE_URL"),
            embedding_dimension = embedder.dimension,
        )

    # Retrieval
    bm25 = BM25Retriever()
    with measure("build_bm25_index", startup_perf) as m:
        m["chunks"] = bm25.load_from_vector_store(store)

    with measure("load_reranker", startup_perf):
        reranker = CrossEncoderReranker(
            model_name = os.getenv("RERANKER_MODEL",
                                   "cross-encoder/ms-marco-MiniLM-L-6-v2")
        )

    domain_scorer = DomainScorer.from_settings()

    retrieval_engine = RetrievalEngine(
        vector_store = store,
        bm25_retriever = bm25,
        embedding_service = embedder,
        reranker = reranker,
        domain_scorer = domain_scorer,
    )
    with measure("domain_centroids", startup_perf) as m:
        retrieval_engine.refresh_domains()
        m["domains"] = sorted(domain_scorer.centroids)

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

    # Build scope from document filenames
    doc_dir = os.environ.get("DOCUMENTS_DIR", "documents")
    if os.path.isdir(doc_dir):
        doc_names = [
            os.path.splitext(f)[0].replace("_", " ")
            for f in os.listdir(doc_dir)
            if f.endswith(".md") and not f.startswith(".")
        ]
        scope = (
            "A knowledge base containing internal company documents about: "
            + ", ".join(doc_names) + ". "
            "A query is in-scope if it could be answered by referencing any of these documents."
        )
    else:
        scope = (
            "A knowledge base containing internal company documents. "
            "A query is in-scope if it relates to any topic that an organization "
            "might document internally. Only reject queries that are clearly "
            "unrelated, like jokes, math, weather, or celebrity gossip."
        )

    # Full pipeline with guardrails
    llm_classifier = LLMClassifier(scope_description=scope)
    with measure("scope_terms", startup_perf) as m:
        keyword_classifier = keyword_classifier_from_settings(doc_dir)
        m["terms"] = len(keyword_classifier.in_scope_terms)
    classifier = QueryClassifierPipeline(keyword_classifier=keyword_classifier,
                                         llm_classifier=llm_classifier)

    rag_pipeline = RAGPipeline(enhanced, generator, detector)
    guarded_pipeline = GuardedRAGPipeline(rag_pipeline, classifier=classifier)

    app_state["pipeline"] = guarded_pipeline
    app_state["store"] = store
    app_state["embedder"] = embedder
    app_state["bm25"] = bm25
    app_state["retrieval_engine"] = retrieval_engine

    # Response cache: keyed on query + index version + pipeline signature
    settings = get_settings()
    app_state["response_cache"] = ResponseCache(
        store = store,
        pipeline_signature = _pipeline_signature(generator, rag_pipeline, domain_scorer,
                                                 llm_classifier, keyword_classifier),
        max_entries = settings.cache_max_entries,
        ttl_seconds = settings.cache_ttl_seconds,
        enabled = settings.cache_enabled,
    )

    startup_perf["rss_ready_mb"] = rss_mb()
    startup_perf["process_peak_mb"] = peak_rss_mb()
    app_state["startup_perf"] = startup_perf

    print(f"Groq API keys loaded: {len(GROQ_KEYS)}")
    print(f"Startup perf: { {k: v.get('ms') for k, v in startup_perf.items() if isinstance(v, dict)} } ms, "
          f"RSS {startup_perf['rss_ready_mb']} MB")
    print("Startup complete - ready to serve")

    yield   # app runs here

    # Shutdown
    store.close()
    print("Shutdown complete")


def _pipeline_signature(generator, rag_pipeline, domain_scorer, classifier, 
                        keyword_classifier = None) -> str:
    """Fingerprint of everything that shapes an answer.

    If the prompt, model, gate threshold or scoping settings change,
    the signature changes and cached answers are no longer served.
    """
    from src.cache import make_key
    return make_key(
        "pipeline-v1",
        getattr(generator, "model", None),
        generator._build_system_prompt(),
        rag_pipeline.gate.min_score,
        (domain_scorer.enabled, domain_scorer.penalty, domain_scorer.min_margin,
         sorted(domain_scorer._alias_patterns)),
        getattr(classifier, "model", None),
        getattr(classifier, "scope_description", None),
        sorted(getattr(keyword_classifier, "in_scope_terms", [])),
        sorted(getattr(keyword_classifier, "strong_terms", [])),
        getattr(keyword_classifier, "min_matches", None),
    )[:16]

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
def chat(request: ChatRequest, api_key: str = Security(verify_api_key)):
    pipeline = app_state["pipeline"]
    cache = app_state.get("response_cache")
    with start_trace("chat", top_k=request.top_k) as trace:
        hit = False
        if cache is not None and request.use_cache:
            with span("cache.response_lookup") as attrs:
                hit, cached, age = cache.get(request.query, request.top_k, request.strategy)
                attrs["hit"] = hit
        if hit:
            result = cached
            result["cache"] = {"hit": True, "age_s": age,
                               "original_trace_id": result.pop("_original_trace_id", None)}
            trace.set(decision="cache_hit", original_trace_id=result["cache"]["original_trace_id"])
            get_trace_store().incr("response_cache_hits")
        else:
            for attempt in range(3):
                try:
                    result = pipeline.answer(query = request.query, top_k = request.top_k)
                    break
                except Exception as e:
                    if "rate" in str(e).lower() or "429" in str(e) or "quota" in str(e).lower():
                        trace.set(rate_limit_retries=attempt + 1)
                        swap_groq_key()
                        time.sleep(5)
                        continue
                    raise
            else:
                trace.set(decision="rate_limited")
                raise HTTPException(status_code=429, detail="All API keys exhausted")
            if cache is not None:
                result["trace_id"] = trace.trace_id
                stored = request.use_cache and cache.put(
                    request.query, request.top_k, request.strategy, result)
                result["cache"] = {"hit": False, "stored": bool(stored)}
    result["answer"] = html.escape(result["answer"])  # Sanitize output to prevent XSS
    result["trace_id"] = trace.trace_id
    result["perf"] = trace.summary()        # final numbers, trace is finished
    return JSONResponse(jsonable_encoder(result), headers={"X-Trace-Id": trace.trace_id})


@app.get("/api/traces")
async def list_traces(limit: int = 20, kind: str | None = None,
                      api_key: str = Security(verify_api_key)):
    """Most recent traces (newest first): id, total time, decision, tokens."""
    limit = max(1, min(limit, 200))
    return {"traces": get_trace_store().recent(limit=limit, kind=kind)}


@app.get("/api/traces/{trace_id}")
async def get_trace(trace_id: str, api_key: str = Security(verify_api_key)):
    """Full trace: every stage with its time and memory, retrieval scores, tokens."""
    trace = get_trace_store().get(trace_id)
    if trace is None:
        raise HTTPException(404, "Trace not found (only the most recent traces are kept in memory)")
    return trace

@app.get("/api/documents")
def documents(api_key: str = Security(verify_api_key)):
    """Ingested documents: version hash, domain, title, chunks kept, duplicates dropped."""
    return {"documents": app_state["store"].list_documents()}

@app.get("/api/metrics")
async def metrics(kind: str | None = None, api_key: str = Security(verify_api_key)):
    """Latency percentiles (overall and per stage), memory, tokens, decisions."""
    report = get_trace_store().metrics(kind=kind)
    report["startup"] = app_state.get("startup_perf", {})
    for name in ("retrieval_engine", "response_cache"):
        component = app_state.get(name)
        stats = getattr(component, "cache_stats", None)
        if callable(stats):
            report.setdefault("caches", {})[name] = stats()
    return report


@app.post("/api/ingest")
def ingest(request: IngestRequest, api_key: str = Security(verify_api_key)):
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

    _after_ingest()

    return {"status": "ok", "files_ingested": len(results), "details": results,
            "perf": getattr(pipeline, "last_perf", None)}


def _after_ingest():
    """Hook for components that must refresh when the knowledge base changes."""
    for name in ("retrieval_engine", "response_cache"):
        component = app_state.get(name)
        refresh = getattr(component, "on_index_changed", None)
        if callable(refresh):
            refresh()

@app.post("/api/evaluate")
def evaluate(request: dict, api_key: str = Security(verify_api_key)):
    """Admin endpoint - run RAGAS evaluation."""

    label = request.get("label", "default")
    dataset_path = request.get("dataset", None)

    # Adversarial suite: deterministic checks through the guarded
    # (production) pipeline; costs generation calls but no judge calls.
    if request.get("suite") == "adversarial":
        from src.evaluation.adversarial import AdversarialRunner, load_cases
        cases = load_cases(dataset_path or "datasets/adversarial_eval.json")
        runner = AdversarialRunner(app_state["pipeline"], cases,
                                   delay_seconds=float(request.get("delay_seconds", 4.0)))
        report = runner.run(label)
        AdversarialRunner.save(report)
        return report

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