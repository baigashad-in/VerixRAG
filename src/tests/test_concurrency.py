"""Tests that the API serves requests concurrently."""

import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.tests import fakes
from src.tests.conftest import FIXTURE_KB

from src.api import main
from src.storage.vector_store import VectorStore
from fastapi.testclient import TestClient
from src.config import reset_settings
from src.generation.generator import RAGGenerator
from src.guardrails.query_classifier import LLMClassifier
from src.ingestion.ingest import IngestionPipeline
from src.observability.tracing import TraceStore, set_trace_store

LLM_SECONDS = 0.5
REQUESTS = 4

@pytest.mark.db
def test_vector_store_uses_one_connection_per_thread(clean_db):
    store = VectorStore(connection_string=clean_db, embedding_dimension=8)
    seen = {}

    def grab(name):
        seen[name] = store.conn
        with store.conn.cursor() as cur:
            cur.execute("SELECT 1")

    threads = [threading.Thread(target=grab, args=(n,)) for n in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen["a"] is not seen["b"]
    assert store.conn is store.conn                     # stable within a thread
    store.close()
    assert seen["a"].closed and seen["b"].closed


@pytest.mark.db
def test_chat_requests_run_concurrently(clean_db, fake_models, tmp_path, monkeypatch):
    shutil.copytree(FIXTURE_KB, tmp_path / "documents")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", clean_db)
    monkeypatch.setenv("API_KEY", "k")
    monkeypatch.setenv("DOMAINS_CONFIG", str(tmp_path / "documents" / "_domains.json"))
    reset_settings()
    set_trace_store(TraceStore(maxlen=100, log_path=""))
    IngestionPipeline(db_connection=clean_db).ingest_directory("./documents")

    def slow_llm(self, messages):
        time.sleep(LLM_SECONDS)                          # stands in for a Groq round trip
        return fakes.fake_completion("Express shipping costs $15 [Source 1].")

    monkeypatch.setattr(RAGGenerator, "_call_with_swap", slow_llm)
    monkeypatch.setattr(LLMClassifier, "classify", lambda self, q: {
        "in_scope": True, "confidence": "high", "reason": "test"})

    main.app_state.clear()
    with TestClient(main.app, base_url="http://localhost") as client:
        def ask(i):
            return client.post("/api/chat", headers={"X-API-Key": "k"}, json={
                "query": f"How much does express shipping cost? ({i})", "use_cache": False})

        start = time.perf_counter()
        with ThreadPoolExecutor(REQUESTS) as pool:
            responses = list(pool.map(ask, range(REQUESTS)))
        elapsed = time.perf_counter() - start

    assert all(r.status_code == 200 for r in responses)
    # Serial handling would take >= REQUESTS * LLM_SECONDS (2.0 s)
    assert elapsed < REQUESTS * LLM_SECONDS * 0.6, f"requests ran serially ({elapsed:.2f}s)"