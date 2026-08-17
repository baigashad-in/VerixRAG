"""
Offline end-to-end tests of the RAG pipeline (no DB, no LLM, no models).

Retrieval runs on the deterministic fakes; the LLM call is replaced by
a canned response. These tests check the plumbing added around the
pipeline: tracing, timings, memory, tokens, and the API endpoints.
"""

import pytest

from src.tests import fakes
from src.observability.tracing import TraceStore, set_trace_store

DOCS = [
    ("To create an account you need a password with at least 8 characters "
     "including one uppercase letter, one number, and one special character.",
     {"source": "account_guide.md", "filename": "account_guide.md", "domain": "store"}),
    ("Express shipping is available for an additional $15 and delivers within "
     "2 business days.",
     {"source": "shipping_policy.md", "filename": "shipping_policy.md", "domain": "store"}),
    ("CloudDeck support phone: 1-888-555-DECK (Enterprise only, 24/7).",
     {"source": "product_manual.md", "filename": "product_manual.md", "domain": "clouddeck"}),
]


@pytest.fixture
def trace_store():
    store = TraceStore(maxlen=50, log_path="")
    set_trace_store(store)
    return store


@pytest.fixture
def pipeline(fake_models, monkeypatch):
    from src.query.enhanced_retrieval import EnhancedRetriever
    from src.generation.generator import RAGGenerator
    from src.generation.hallucination_check import HallucinationDetector
    from src.pipeline import RAGPipeline

    engine, store, embedder = fakes.build_engine(DOCS)
    enhanced = EnhancedRetriever(engine, query_transformer=None, embedding_service=embedder)
    generator = RAGGenerator()
    monkeypatch.setattr(generator, "_call_with_swap", lambda messages: fakes.fake_completion(
        "Passwords need at least 8 characters with an uppercase letter, a number "
        "and a special character [Source 1]."))
    return RAGPipeline(enhanced, generator, HallucinationDetector())


def test_pipeline_returns_perf_and_trace(pipeline, trace_store):
    out = pipeline.answer("What are the password requirements?", top_k=2)

    perf = out["perf"]
    assert perf["trace_id"] == out["trace_id"]
    assert perf["total_ms"] is not None
    assert {"retrieval", "generation", "hallucination_check"} <= set(perf["stages_ms"])
    assert perf["tokens"] == {"prompt": 1200, "completion": 60, "total": 1260}
    assert perf["memory_mb"]["process_peak"] > 0
    assert out["usage"]["total_tokens"] == 1260

    trace = trace_store.get(out["trace_id"])
    span_names = {s["name"] for s in trace["spans"]}
    assert {"retrieval.embed_query", "retrieval.hybrid_search", "dense_search",
            "sparse_search", "retrieval.rerank", "generation.llm",
            "generation.enforce_citations"} <= span_names
    assert trace["attrs"]["decision"] == "answered"
    assert trace["attrs"]["retrieved"][0]["source"] == "account_guide.md"
    assert "rerank_score" in trace["attrs"]["retrieved"][0]


def test_api_chat_traces_and_metrics(pipeline, trace_store, monkeypatch):
    from fastapi.testclient import TestClient
    from src.api import main
    from src.guardrails.guardrails_pipeline import GuardedRAGPipeline

    class AllowAll:
        def classify(self, query):
            return {"in_scope": True, "confidence": "high", "reason": "test", "stage": "keyword"}

    monkeypatch.setenv("API_KEY", "test-key")
    main.app_state.clear()
    main.app_state["pipeline"] = GuardedRAGPipeline(pipeline, classifier=AllowAll())
    main.app_state["startup_perf"] = {"load_embedder": {"ms": 1.0}}
    # no `with`: skip lifespan model loading; localhost passes TrustedHostMiddleware
    client = TestClient(main.app, base_url="http://localhost")
    headers = {"X-API-Key": "test-key"}

    resp = client.post("/api/chat", json={"query": "password requirements"}, headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    trace_id = resp.headers["X-Trace-Id"]
    assert body["trace_id"] == trace_id == body["perf"]["trace_id"]
    assert body["perf"]["total_ms"] > 0
    assert "guardrails.classify" in body["perf"]["stages_ms"]

    full = client.get(f"/api/traces/{trace_id}", headers=headers).json()
    assert full["kind"] == "chat" and full["attrs"]["decision"] == "answered"
    assert any(s["name"] == "generation.llm" for s in full["spans"])

    listing = client.get("/api/traces", headers=headers).json()["traces"]
    assert listing[0]["trace_id"] == trace_id

    metrics = client.get("/api/metrics", headers=headers).json()
    assert metrics["window"]["traces"] >= 1
    assert metrics["decisions"]["answered"] >= 1
    assert metrics["startup"]["load_embedder"]["ms"] == 1.0

    assert client.get("/api/traces/nope", headers=headers).status_code == 404
    assert client.get("/api/metrics").status_code in (401, 403)


class StubMetrics:
    """Judge stand-in: fixed scores, no LLM calls."""
    def __init__(self):
        self.trace, self.last_claims, self.tokens_used = [], None, 0

    def faithfulness(self, answer, contexts):
        self.last_claims = (1, 1)
        return 1.0

    def answer_relevancy(self, q, a):
        return 0.8

    def context_precision(self, q, contexts, gt):
        return 1.0

    def context_recall(self, gt, contexts):
        return 1.0


def test_eval_report_includes_performance_and_categories(pipeline, trace_store):
    from src.evaluation.eval_dataset import EvalDataset
    from src.evaluation.ragas_eval import EvaluationRunner

    ds = EvalDataset()
    ds.add("What are the password requirements?", "8 characters...", [], category="easy")
    ds.add("How fast is express shipping?", "2 business days", [], category="medium")
    report = EvaluationRunner(pipeline, StubMetrics(), ds).run("offline")

    perf = report["performance"]
    assert perf["pipeline_latency_ms"]["n"] == 2
    assert "retrieval" in perf["stage_latency_ms"]
    assert perf["generation_tokens_per_question"]["mean"] == 1260
    assert perf["memory_mb"]["process_peak"] > 0
    assert set(report["by_category"]) == {"easy", "medium"}
    assert report["per_example"][0]["perf"]["wall_ms"] > 0
    assert report["contexts_re_retrieved"] == 0
