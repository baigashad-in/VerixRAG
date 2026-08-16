"""Tests for the retrieval confidence gate."""

from types import SimpleNamespace

import pytest

from src.retrieval.confidence import ABSTAIN_ANSWER, ConfidenceGate
from src.tests import fakes
from src.tests.test_pipeline_offline import DOCS


def results(*scores):
    return [SimpleNamespace(rerank_score=s) for s in scores]


def test_blocks_when_best_score_below_threshold():
    d = ConfidenceGate(-4.0).check(results(-9.0, -5.5))
    assert not d.passed and d.top_score == -5.5 and d.threshold == -4.0


def test_passes_at_or_above_threshold():
    assert ConfidenceGate(-4.0).check(results(-4.0, -9)).passed
    assert ConfidenceGate(-4.0).check(results(2.5)).passed


def test_disabled_gate_always_passes():
    d = ConfidenceGate(None).check(results(-11))
    assert d.passed and d.threshold is None


def test_no_rerank_scores_passes():
    d = ConfidenceGate(-4.0).check([SimpleNamespace(rrf_score=0.03)])
    assert d.passed and d.top_score is None


def test_uses_raw_score_not_domain_adjusted():
    r = SimpleNamespace(rerank_score=1.0, final_score=-3.0)
    assert ConfidenceGate(0.0).check([r]).passed


def test_threshold_read_from_env(monkeypatch):
    from src.config import get_settings, reset_settings
    monkeypatch.setenv("RERANK_MIN_SCORE", "-2.5")
    reset_settings()
    assert get_settings().rerank_min_score == -2.5
    monkeypatch.setenv("RERANK_MIN_SCORE", "off")
    reset_settings()
    assert get_settings().rerank_min_score is None


@pytest.fixture
def pipeline_with_counter(fake_models, monkeypatch):
    from src.query.enhanced_retrieval import EnhancedRetriever
    from src.generation.generator import RAGGenerator
    from src.pipeline import RAGPipeline

    engine, _, embedder = fakes.build_engine(DOCS)
    calls = {"llm": 0}

    def fake_llm(messages):
        calls["llm"] += 1
        return fakes.fake_completion("Express shipping costs $15 [Source 1].")

    gen = RAGGenerator()
    monkeypatch.setattr(gen, "_call_with_swap", fake_llm)
    pipe = RAGPipeline(EnhancedRetriever(engine, None, embedder), gen,
                       confidence_gate=ConfidenceGate(-1.0))
    return pipe, calls


def test_pipeline_abstains_without_llm_call(pipeline_with_counter):
    pipe, calls = pipeline_with_counter
    out = pipe.answer("What is the capital city of France?", top_k=3)
    assert out["answer"] == ABSTAIN_ANSWER
    assert out["abstained"]["passed"] is False
    assert calls["llm"] == 0
    assert "generation" not in out["perf"]["stages_ms"]
    assert "confidence_gate" in out["perf"]["stages_ms"]


def test_pipeline_answers_when_confident(pipeline_with_counter):
    pipe, calls = pipeline_with_counter
    out = pipe.answer("How much does express shipping cost?", top_k=3)
    assert calls["llm"] == 1
    assert out["confidence"]["passed"] is True
    assert "$15" in out["answer"]


def test_calibration_recommendation_respects_loss_limit():
    from scripts.calibrate import recommend, threshold_table
    in_scores = [3.0, 2.0, 1.0, -2.0, -7.5]          # one hard in-scope question
    out_scores = [-10.0, -9.5, -8.0]
    rows = threshold_table(in_scores, out_scores, lo=-11, hi=4, step=0.5)
    rec = recommend(rows, max_loss=0.0)
    assert rec["threshold"] == -7.5 and rec["in_scope_blocked"] == 0
    assert rec["out_of_scope_caught"] == 3
