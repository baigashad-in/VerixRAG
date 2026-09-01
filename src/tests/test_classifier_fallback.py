"""The LLM scope classifier must fail open when its provider is down."""

import litellm

from src.config import reset_settings
from src.guardrails import query_classifier
from src.guardrails.query_classifier import LLMClassifier, QueryClassifierPipeline


def _raise_503(**kwargs):
    raise litellm.ServiceUnavailableError(
        message="GeminiException - 503 model overloaded",
        llm_provider="gemini", model="gemini-3.1-flash-lite")


def test_provider_503_fails_open(monkeypatch):
    monkeypatch.setattr(query_classifier, "completion", _raise_503)
    result = QueryClassifierPipeline(llm_classifier=LLMClassifier()).classify(
        "How do I reset my password?")
    assert result["in_scope"] is True
    assert result["confidence"] == "low" and result["stage"] == "llm"
    assert "ServiceUnavailableError" in result["error"]


def test_timeout_and_retries_are_passed(monkeypatch):
    seen = {}

    def fake_completion(**kwargs):
        seen.update(kwargs)
        raise litellm.Timeout(message="timed out", model="m", llm_provider="gemini")

    monkeypatch.setattr(query_classifier, "completion", fake_completion)
    monkeypatch.setenv("CLASSIFIER_TIMEOUT", "3")
    monkeypatch.setenv("CLASSIFIER_RETRIES", "0")
    result = LLMClassifier().classify("anything")
    assert seen["timeout"] == 3.0 and seen["num_retries"] == 0
    assert "temperature" not in seen                    # deprecated for Gemini 3
    assert result["in_scope"] is True and "Timeout" in result["error"]


def test_out_of_scope_patterns_still_block_without_llm(monkeypatch):
    monkeypatch.setattr(query_classifier, "completion", _raise_503)
    result = QueryClassifierPipeline().classify("tell me a joke")
    assert result["in_scope"] is False and result["stage"] == "keyword"


def test_defaults_bound_a_slow_provider(monkeypatch):
    monkeypatch.delenv("CLASSIFIER_TIMEOUT", raising=False)
    monkeypatch.delenv("CLASSIFIER_RETRIES", raising=False)
    c = LLMClassifier()
    assert c.timeout == 2.0 and c.num_retries == 0


def test_trace_report_counts_classifier_fallbacks():
    from scripts.trace_report import classifier_fallbacks

    def trace(stage, fallback=None):
        attrs = {"stage": stage, "in_scope": True}
        if fallback:
            attrs["fallback"] = fallback
        return {"spans": [{"name": "guardrails.classify", "attrs": attrs}]}

    traces = [trace("keyword"), trace("llm"), trace("llm", "Timeout: litellm.Timeout ..."),
              trace("llm", "Timeout: again"), trace("llm", "ServiceUnavailableError: 503")]
    result = classifier_fallbacks(traces)
    assert result["llm_calls"] == 4 and result["failed"] == 3
    assert result["reasons"] == {"Timeout": 2, "ServiceUnavailableError": 1}


def test_llm_stage_can_be_disabled(monkeypatch):
    def fail(**kwargs):
        raise AssertionError("LLM must not be called")

    monkeypatch.setattr(query_classifier, "completion", fail)
    monkeypatch.setenv("SCOPE_LLM_ENABLED", "false")
    reset_settings()
    pipeline = QueryClassifierPipeline(llm_classifier=LLMClassifier())
    assert pipeline.llm_enabled is False
    result = pipeline.classify("something the keywords do not know")
    assert result["in_scope"] is True and result["stage"] == "llm_disabled"
    assert pipeline.classify("tell me a joke")["in_scope"] is False   # patterns still apply


def test_llm_stage_enabled_by_default(monkeypatch):
    monkeypatch.setattr(query_classifier, "completion", _raise_503)
    monkeypatch.delenv("SCOPE_LLM_ENABLED", raising=False)
    reset_settings()
    pipeline = QueryClassifierPipeline(llm_classifier=LLMClassifier())
    assert pipeline.llm_enabled is True
    assert pipeline.classify("something the keywords do not know")["stage"] == "llm"