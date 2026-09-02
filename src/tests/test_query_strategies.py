"""Query strategies: rewriting changes the candidates, reranking uses the original question."""

import pytest

from src.query import query_transformer
from src.query.enhanced_retrieval import EnhancedRetriever, QueryStrategy
from src.query.query_transformer import QueryTransformer
from src.tests import fakes

DOCS = [
    ("Refunds are processed within 14 days of the return.", {"source": "refund.md"}),
    ("Express shipping costs $15 and arrives in 2 business days.", {"source": "shipping.md"}),
    ("Error E-4012 means the payment card was declined.", {"source": "errors.md"}),
    ("Reset your password from the login page using the email link.", {"source": "account.md"}),
]


class FakeTransformer:
    def __init__(self, fail=False):
        self.fail, self.calls = fail, []

    def _call(self, name, query):
        self.calls.append(name)
        if self.fail:
            raise TimeoutError("rewrite timed out")

    def hyde(self, query):
        self._call("hyde", query)
        return "Express shipping costs $15 per order."

    def multi_query(self, query):
        self._call("multi", query)
        return [query, "express shipping price", "card declined error"]

    def expand_query(self, query):
        self._call("expand", query)
        return query + " delivery courier"


@pytest.fixture
def spied(fake_models):
    engine, _, embedder = fakes.build_engine(DOCS)
    engine.cache = None
    seen = {"search": [], "rerank": []}
    search, rerank = engine.search_candidates, engine.rerank

    def spy_search(text, emb, fetch_k, metadata_filter=None):
        seen["search"].append((text, emb))
        return search(text, emb, fetch_k, metadata_filter)

    def spy_rerank(query, candidates, top_k, query_embedding=None):
        seen["rerank"].append((query, [c.chunk_id for c in candidates]))
        return rerank(query, candidates, top_k, query_embedding)

    engine.search_candidates, engine.rerank = spy_search, spy_rerank
    return engine, embedder, seen


def test_none_skips_the_transformer(spied):
    engine, embedder, seen = spied
    t = FakeTransformer()
    EnhancedRetriever(engine, t, embedder).retrieve("refund time", top_k=2)
    assert t.calls == []


def test_hyde_dense_uses_hypothetical_and_rerank_uses_question(spied):
    engine, embedder, seen = spied
    q = "how much is fast delivery"
    results = EnhancedRetriever(engine, FakeTransformer(), embedder).retrieve(
        q, top_k=2, strategy=QueryStrategy.HYDE)
    text, emb = seen["search"][0]
    assert text == q                                             # BM25 keeps the original words
    assert emb == embedder.embed_text("Express shipping costs $15 per order.")
    assert seen["rerank"][0][0] == q
    assert results and all(r.rerank_score is not None for r in results)


def test_multi_query_unions_candidates_without_duplicates(spied):
    engine, embedder, seen = spied
    q = "shipping cost and declined card"
    EnhancedRetriever(engine, FakeTransformer(), embedder).retrieve(
        q, top_k=3, strategy=QueryStrategy.MULTI_QUERY)
    assert [s[0] for s in seen["search"]] == [q, "express shipping price", "card declined error"]
    query, ids = seen["rerank"][0]
    assert query == q and len(ids) == len(set(ids))


def test_expand_searches_expanded_text_and_reranks_original(spied):
    engine, embedder, seen = spied
    q = "express shipping"
    EnhancedRetriever(engine, FakeTransformer(), embedder).retrieve(
        q, top_k=2, strategy=QueryStrategy.EXPAND)
    assert seen["search"][0][0] == "express shipping delivery courier"
    assert seen["rerank"][0][0] == q


def test_rewrite_failure_falls_back_to_plain_query(spied):
    from src.observability.tracing import TraceStore, start_trace
    engine, embedder, seen = spied
    with start_trace("test", store=TraceStore(maxlen=5, log_path="")) as trace:
        results = EnhancedRetriever(engine, FakeTransformer(fail=True), embedder).retrieve(
            "refund time", top_k=2, strategy=QueryStrategy.HYDE)
    rewrite = [s for s in trace.spans if s["name"] == "retrieval.rewrite"][0]
    assert "TimeoutError" in rewrite["attrs"]["fallback"]
    assert results and seen["search"][0][0] == "refund time"


def test_transformer_kwargs_by_provider(monkeypatch):
    seen = []

    def fake(**kwargs):
        seen.append(kwargs)
        return fakes.fake_completion('["a", "b"]', prompt_tokens=50, completion_tokens=10)

    monkeypatch.setattr(query_transformer, "completion", fake)
    QueryTransformer(model="gemini/gemini-3.1-flash-lite", timeout=3, num_retries=0).hyde("q")
    QueryTransformer(model="groq/openai/gpt-oss-20b").hyde("q")
    assert "temperature" not in seen[0] and seen[0]["timeout"] == 3 and seen[0]["num_retries"] == 0
    assert seen[1]["temperature"] == 0.0 and "api_key" not in seen[1]


def test_transformer_parsing_and_stats(monkeypatch):
    replies = iter(['["sub one", "q", "sub two"]', "not json",
                    '"refund policy return money"\nExplanation: ...'])
    monkeypatch.setattr(query_transformer, "completion",
                        lambda **k: fakes.fake_completion(next(replies), 40, 10))
    t = QueryTransformer(model="groq/x")
    assert t.multi_query("q") == ["q", "sub one", "sub two"]
    assert t.multi_query("q") == ["q"]                          # malformed reply: original only
    assert t.expand_query("refund policy") == "refund policy return money"
    assert t.stats["calls"] == 3 and t.stats["tokens"] == 150 and t.stats["errors"] == 0


def test_transformer_counts_errors(monkeypatch):
    def boom(**kwargs):
        raise TimeoutError("slow")

    monkeypatch.setattr(query_transformer, "completion", boom)
    t = QueryTransformer(model="groq/x")
    with pytest.raises(TimeoutError):
        t.hyde("q")
    assert t.stats["calls"] == 1 and t.stats["errors"] == 1


def test_api_passes_strategy_to_pipeline(fake_models, monkeypatch):
    from fastapi.testclient import TestClient
    from src.api import main
    from src.cache.response_cache import ResponseCache
    from src.generation.generator import RAGGenerator
    from src.guardrails.guardrails_pipeline import GuardedRAGPipeline
    from src.observability.tracing import TraceStore, set_trace_store
    from src.pipeline import RAGPipeline

    set_trace_store(TraceStore(maxlen=20, log_path=""))
    engine, store, embedder = fakes.build_engine(DOCS)
    t = FakeTransformer()
    gen = RAGGenerator()
    monkeypatch.setattr(gen, "_call_with_swap",
                        lambda m: fakes.fake_completion("Express shipping costs $15 [Source 1]."))

    class AllowAll:
        def classify(self, query):
            return {"in_scope": True, "confidence": "high", "reason": "t", "stage": "keyword"}

    monkeypatch.setenv("API_KEY", "k")
    main.app_state.clear()
    main.app_state["pipeline"] = GuardedRAGPipeline(
        RAGPipeline(EnhancedRetriever(engine, t, embedder), gen), classifier=AllowAll())
    main.app_state["response_cache"] = ResponseCache(store, "sig")
    c = TestClient(main.app, base_url="http://localhost")
    r = c.post("/api/chat", json={"query": "How much is express shipping?", "strategy": "hyde",
                                  "use_cache": False}, headers={"X-API-Key": "k"}).json()
    assert r["query_strategy"] == "hyde" and t.calls == ["hyde"]


@pytest.mark.db
def test_retrieval_eval_runs_strategies(clean_db, fake_models, tmp_path, monkeypatch):
    import json
    import shutil
    import sys

    from src.config import reset_settings
    from src.ingestion.ingest import IngestionPipeline
    from src.tests.conftest import FIXTURE_KB

    kb = tmp_path / "documents"
    shutil.copytree(FIXTURE_KB, kb)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", clean_db)
    monkeypatch.setenv("DOMAINS_CONFIG", str(kb / "_domains.json"))
    monkeypatch.setenv("QUERY_REWRITE_MODEL", "groq/fake")
    reset_settings()
    IngestionPipeline(db_connection=clean_db, documents_root=str(kb)).ingest_directory(str(kb))
    monkeypatch.setattr(query_transformer, "completion", lambda **k: fakes.fake_completion(
        '["express shipping cost", "password reset link"]', 30, 10))

    dataset = tmp_path / "eval.json"
    dataset.write_text(json.dumps({"questions": [
        {"question": "How much does express shipping cost?", "category": "factual",
         "source_file": "shipping_policy.md", "sections": ["Express Shipping"]},
        {"question": "How long is the password reset link valid?", "category": "factual",
         "source_file": "account_guide.md", "sections": ["Password Reset"]}]}))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["retrieval_eval", "--dataset", str(dataset), "--label", "s",
                                      "--output-dir", str(out), "--configs", "full,multi,hyde,expand"])
    from scripts import retrieval_eval
    retrieval_eval.main()

    report = json.loads(next(out.glob("retrieval_eval_s_*.json")).read_text())
    assert list(report["configs"]) == ["full", "multi", "hyde", "expand"]
    assert "rewrite" not in report["configs"]["full"]
    for name in ("multi", "hyde", "expand"):
        rw = report["configs"][name]["rewrite"]
        assert rw["calls"] == 2 and rw["errors"] == 0 and rw["tokens_per_question"] == 40
        assert 0.0 <= report["configs"][name]["metrics"]["mrr"] <= 1.0
        assert "factual" in report["configs"][name]["by_category"]