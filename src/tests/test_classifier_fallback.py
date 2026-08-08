"""The LLM scope classifier must fail open when its provider is down."""

import litellm

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
    assert result["in_scope"] is True and "Timeout" in result["error"]


def test_out_of_scope_patterns_still_block_without_llm(monkeypatch):
    monkeypatch.setattr(query_classifier, "completion", _raise_503)
    result = QueryClassifierPipeline().classify("tell me a joke")
    assert result["in_scope"] is False and result["stage"] == "keyword"