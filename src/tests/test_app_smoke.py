"""
Full-app smoke test: real FastAPI lifespan + real pgvector, fake models/LLM.

Exercises the startup wiring in src/api/main.py (domain scorer, caches,
startup perf, pipeline signature) that unit tests don't reach.
Skipped unless TEST_DATABASE_URL points at Postgres with pgvector.
"""

import shutil

import pytest

from src.tests import fakes
from src.tests.conftest import FIXTURE_KB


@pytest.mark.db
def test_app_end_to_end(clean_db, fake_models, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from src.config import reset_settings
    from src.ingestion.ingest import IngestionPipeline
    from src.guardrails.query_classifier import LLMClassifier
    from src.generation.generator import RAGGenerator
    from src.observability.tracing import TraceStore, set_trace_store

    shutil.copytree(FIXTURE_KB, tmp_path / "documents")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", clean_db)
    monkeypatch.setenv("API_KEY", "smoke")
    monkeypatch.setenv("DOMAINS_CONFIG", str(tmp_path / "documents" / "_domains.json"))
    monkeypatch.setenv("RERANK_MIN_SCORE", "-5.5")
    reset_settings()
    set_trace_store(TraceStore(maxlen=100, log_path=""))

    ingest = IngestionPipeline(db_connection=clean_db)
    ingest.ingest_directory("./documents")
    ingest.store.close()

    llm_calls = {"n": 0}

    def fake_llm(self, messages):
        llm_calls["n"] += 1
        return fakes.fake_completion("Call 1-800-555-0199 for store refunds [Source 1].")

    monkeypatch.setattr(RAGGenerator, "_call_with_swap", fake_llm)
    monkeypatch.setattr(LLMClassifier, "classify", lambda self, q: {
        "in_scope": "capital" not in q.lower(), "confidence": "high", "reason": "smoke"})

    from src.api import main
    main.app_state.clear()
    with TestClient(main.app, base_url="http://localhost") as client:
        h = {"X-API-Key": "smoke"}

        # Startup recorded model/index load cost and domain centroids
        metrics = client.get("/api/metrics", headers=h).json()
        startup = metrics["startup"]
        assert {"load_embedder", "load_reranker", "build_bm25_index", "domain_centroids"} <= set(startup)
        assert set(startup["domain_centroids"]["domains"]) == {"store", "clouddeck", "verixapi", "hr"}
        assert startup["rss_ready_mb"] > 0

        # A store question: answered, scoped to the store domain, traced
        q = {"query": "What VerixStore phone number do I call about a refund?"}
        first = client.post("/api/chat", json=q, headers=h).json()
        assert "1-800-555-0199" in first["answer"]
        assert first["scoping"]["domain"] == "store"
        assert first["confidence"]["passed"] is True
        assert first["citations"][0]["section"]                 # section path in citations
        assert first["cache"] == {"hit": False, "stored": True}

        # Same question again: response cache, no LLM call
        second = client.post("/api/chat", json=q, headers=h).json()
        assert second["cache"]["hit"] is True and llm_calls["n"] == 1

        # Out-of-scope: blocked by guardrails before retrieval
        oos = client.post("/api/chat", json={"query": "What is the capital of France?"},
                          headers=h).json()
        assert oos["guardrails"]["classified_as"] == "out_of_scope"

        # Document registry
        docs = {d["source"]: d for d in client.get("/api/documents", headers=h).json()["documents"]}
        assert docs["product_manual.md"]["domain"] == "clouddeck"
        assert docs["policies.md"]["duplicates_dropped"] >= 1

        # Re-ingest through the API: nothing changed, caches refreshed
        resp = client.post("/api/ingest", json={"directory": "./documents"}, headers=h).json()
        assert all(d["status"] == "unchanged" for d in resp["details"])
        assert resp["perf"]["documents_unchanged"] == 7

        trace = client.get(f"/api/traces/{first['trace_id']}", headers=h).json()
        names = {s["name"] for s in trace["spans"]}
        classify = next(s for s in trace["spans"] if s["name"] == "guardrails.classify")
        assert classify["attrs"]["stage"] == "keyword"          # seeded terms skip the LLM
        assert startup["scope_terms"]["terms"] > 0
        assert {"guardrails.classify", "retrieval.rerank", "retrieval.scoping",
                "confidence_gate", "generation.llm"} <= names
