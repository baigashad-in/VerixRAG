"""Tests for the retrieval and response caches."""

import time

import pytest

from src.cache import TTLCache, make_key, normalize_query
from src.cache.response_cache import ResponseCache
from src.tests import fakes
from src.tests.test_pipeline_offline import DOCS


class Versioned:
    def __init__(self):
        self.v = "1"

    def index_version(self):
        return self.v


# ── TTLCache ──

def test_hit_miss_and_stats():
    c = TTLCache(max_entries=2, ttl_seconds=60)
    assert c.get("a")[0] is False
    c.set("a", {"x": 1})
    hit, value, age = c.get("a")
    assert hit and value == {"x": 1} and age >= 0
    assert c.stats()["hits"] == 1 and c.stats()["misses"] == 1


def test_values_are_isolated_copies():
    c = TTLCache()
    original = {"answer": "x"}
    c.set("k", original)
    original["answer"] = "mutated"
    _, got, _ = c.get("k")
    got["answer"] = "changed by caller"
    assert c.get("k")[1]["answer"] == "x"


def test_lru_eviction():
    c = TTLCache(max_entries=2)
    c.set("a", 1); c.set("b", 2)
    c.get("a")                         # a is now most recent
    c.set("c", 3)                      # evicts b
    assert c.get("b")[0] is False and c.get("a")[0] and c.get("c")[0]
    assert c.stats()["evictions"] == 1


def test_ttl_expiry():
    c = TTLCache(ttl_seconds=0.01)
    c.set("a", 1)
    time.sleep(0.02)
    assert c.get("a")[0] is False and c.stats()["expired"] == 1


def test_query_normalisation_and_keys():
    assert normalize_query("  What is X?? ") == normalize_query("what   is x")
    assert make_key("a", 1) == make_key("a", 1) != make_key("a", 2)


# ── ResponseCache ──

def test_response_cache_respects_index_version_and_signature():
    store = Versioned()
    rc = ResponseCache(store, pipeline_signature="sig1")
    rc.put("What is X?", 8, "none", {"answer": "X is y.", "trace_id": "t1",
                                    "guardrails": {"input_pii_detected": 0}})
    hit, value, _ = rc.get("what is x", 8, "none")
    assert hit and value["answer"] == "X is y." and value["_original_trace_id"] == "t1"
    assert "trace_id" not in value

    store.v = "2"                                   # documents re-ingested
    assert rc.get("what is x", 8, "none")[0] is False

    rc2 = ResponseCache(store, pipeline_signature="sig2")   # prompt/model changed
    rc.put("q", 8, "none", {"answer": "a"})
    assert rc2.get("q", 8, "none")[0] is False


def test_pii_and_blocked_answers_not_cached():
    rc = ResponseCache(Versioned(), "sig")
    assert not rc.put("my ssn is 123-45-6789", 8, "none",
                      {"answer": "a", "guardrails": {"input_pii_detected": 1}})
    assert not rc.put("q", 8, "none", {"answer": "a", "guardrails": {"original_blocked": True}})
    assert not rc.put("q", 8, "none", {"answer": ""})


def test_disabled_response_cache():
    rc = ResponseCache(Versioned(), "sig", enabled=False)
    assert not rc.put("q", 8, "none", {"answer": "a"})
    assert rc.get("q", 8, "none")[0] is False


# ── Retrieval cache in the engine ──

def test_engine_cache_hits_and_invalidates(fake_models):
    engine, store, embedder = fakes.build_engine(DOCS)
    calls = {"n": 0}
    original = embedder.embed_text

    def counting(text):
        calls["n"] += 1
        return original(text)
    embedder.embed_text = counting

    first = engine.retrieve("express shipping cost", top_k=2)
    second = engine.retrieve("Express shipping cost?", top_k=2)
    assert calls["n"] == 1                         # second call served from cache
    assert [r.chunk_id for r in first] == [r.chunk_id for r in second]
    assert engine.cache_stats()["hits"] == 1

    store.add("Express shipping now costs $20.", {"source": "new.md"},
              original("Express shipping now costs $20."))   # index version changes
    engine.retrieve("express shipping cost", top_k=2)
    assert calls["n"] == 2

    engine.on_index_changed()
    assert engine.cache_stats()["entries"] == 0


def test_cache_disabled_by_setting(fake_models, monkeypatch):
    from src.config import reset_settings
    monkeypatch.setenv("CACHE_ENABLED", "false")
    reset_settings()
    engine, _, _ = fakes.build_engine(DOCS)
    assert engine.cache is None and engine.cache_stats() is None


# ── API: response cache end to end ──

@pytest.fixture
def client(fake_models, monkeypatch):
    from fastapi.testclient import TestClient
    from src.api import main
    from src.query.enhanced_retrieval import EnhancedRetriever
    from src.generation.generator import RAGGenerator
    from src.pipeline import RAGPipeline
    from src.guardrails.guardrails_pipeline import GuardedRAGPipeline
    from src.observability.tracing import TraceStore, set_trace_store

    set_trace_store(TraceStore(maxlen=50, log_path=""))
    engine, store, embedder = fakes.build_engine(DOCS)
    calls = {"llm": 0}

    def fake_llm(messages):
        calls["llm"] += 1
        return fakes.fake_completion("Express shipping costs $15 [Source 1].")

    gen = RAGGenerator()
    monkeypatch.setattr(gen, "_call_with_swap", fake_llm)
    rag = RAGPipeline(EnhancedRetriever(engine, None, embedder), gen)

    class AllowAll:
        def classify(self, query):
            return {"in_scope": True, "confidence": "high", "reason": "t", "stage": "keyword"}

    monkeypatch.setenv("API_KEY", "k")
    main.app_state.clear()
    main.app_state["pipeline"] = GuardedRAGPipeline(rag, classifier=AllowAll())
    main.app_state["retrieval_engine"] = engine
    main.app_state["response_cache"] = ResponseCache(store, "sig")
    return TestClient(main.app, base_url="http://localhost"), calls


def test_api_second_identical_query_skips_llm(client):
    c, calls = client
    h = {"X-API-Key": "k"}
    first = c.post("/api/chat", json={"query": "How much is express shipping?"}, headers=h).json()
    second = c.post("/api/chat", json={"query": "how much is express shipping"}, headers=h).json()
    assert calls["llm"] == 1
    assert first["cache"] == {"hit": False, "stored": True}
    assert second["cache"]["hit"] is True
    assert second["cache"]["original_trace_id"] == first["trace_id"]
    assert second["trace_id"] != first["trace_id"]
    assert second["answer"] == first["answer"]               # escaped exactly once
    assert "generation" not in second["perf"]["stages_ms"]

    fresh = c.post("/api/chat", json={"query": "How much is express shipping?",
                                      "use_cache": False}, headers=h).json()
    assert calls["llm"] == 2 and fresh["cache"]["hit"] is False

    metrics = c.get("/api/metrics", headers=h).json()
    assert metrics["caches"]["response_cache"]["hits"] == 1
    assert metrics["caches"]["retrieval_engine"]["hits"] >= 1
    assert metrics["decisions"]["cache_hit"] == 1


def test_api_pii_query_not_cached(client):
    c, calls = client
    h = {"X-API-Key": "k"}
    q = {"query": "My SSN is 123-45-6789, how much is express shipping?"}
    c.post("/api/chat", json=q, headers=h)
    again = c.post("/api/chat", json=q, headers=h).json()
    assert calls["llm"] == 2 and again["cache"]["hit"] is False


@pytest.mark.db
def test_vector_store_index_version_changes_on_write(clean_db):
    from types import SimpleNamespace
    from src.storage.vector_store import VectorStore

    store = VectorStore(clean_db, embedding_dimension=4)
    v0 = store.index_version()
    assert store.index_version() == v0                     # memoised
    chunk = SimpleNamespace(content="hello", embedding=[0.1, 0.2, 0.3, 0.4],
                            metadata={"source": "a.md", "domain": "store"})
    store.replace_source("a.md", [chunk])
    v1 = store.index_version()
    assert v1 != v0
    store.replace_source("a.md", [chunk])                  # same count, new ids
    assert store.index_version() != v1
    store.close()
