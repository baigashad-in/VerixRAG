"""Tests for Groq key rotation and rate-limit handling in the generator."""

from types import SimpleNamespace

import pytest

from src.generation import generator as gen_module
from src.generation.generator import RAGGenerator, _retry_after
from src.observability.tracing import TraceStore, start_trace
from src.tests import fakes


class RateLimited(Exception):
    def __init__(self, retry_after=None, message="Error code: 429 - rate limit reached"):
        super().__init__(message)
        self.response = SimpleNamespace(headers={"retry-after": retry_after} if retry_after else {})


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setattr(RAGGenerator, "_groq_keys", ["k1", "k2", "k3"])
    monkeypatch.setattr(RAGGenerator, "_key_counter", 0)
    sleeps = []
    monkeypatch.setattr(gen_module.time, "sleep", sleeps.append)
    return sleeps


def _completion(monkeypatch, behaviour):
    used = []

    def fake(model, messages, temperature, api_key=None):
        used.append(api_key)
        outcome = behaviour(api_key, len(used))
        if isinstance(outcome, Exception):
            raise outcome
        return fakes.fake_completion("ok [Source 1].")

    monkeypatch.setattr(gen_module, "completion", fake)
    return used


def test_calls_rotate_across_keys(keys, monkeypatch):
    used = _completion(monkeypatch, lambda key, n: None)
    gen = RAGGenerator()
    for _ in range(6):
        gen._call_with_swap([{"role": "user", "content": "q"}])
    assert used == ["k1", "k2", "k3", "k1", "k2", "k3"]


def test_rate_limit_moves_to_next_key_without_waiting(keys, monkeypatch):
    used = _completion(monkeypatch, lambda key, n: RateLimited() if key == "k1" else None)
    RAGGenerator()._call_with_swap([{"role": "user", "content": "q"}])
    assert used == ["k1", "k2"] and keys == []


def test_waits_only_after_every_key_refused_using_retry_after(keys, monkeypatch):
    used = _completion(monkeypatch, lambda key, n: RateLimited("1.5") if n <= 3 else None)
    store = TraceStore(maxlen=5, log_path="")
    with start_trace("chat", store=store) as trace:
        RAGGenerator()._call_with_swap([{"role": "user", "content": "q"}])
    assert used == ["k1", "k2", "k3", "k1"]
    assert keys == [1.5]                                     # one wait, as long as asked
    saved = store.get(trace.trace_id)
    assert saved["attrs"]["llm_rate_limited"] == 3
    assert "generation.rate_limit_wait" in {s["name"] for s in saved["spans"]}


def test_gives_up_after_max_rounds(keys, monkeypatch):
    _completion(monkeypatch, lambda key, n: RateLimited())
    with pytest.raises(RateLimited):
        RAGGenerator()._call_with_swap([{"role": "user", "content": "q"}], max_rounds=2)
    assert len(keys) == 1                                    # waited between rounds, not after the last


def test_other_errors_are_not_retried(keys, monkeypatch):
    used = _completion(monkeypatch, lambda key, n: ValueError("bad request"))
    with pytest.raises(ValueError):
        RAGGenerator()._call_with_swap([{"role": "user", "content": "q"}])
    assert len(used) == 1


def test_retry_after_parsing():
    assert _retry_after(RateLimited("2"), default=9) == 2.0
    assert _retry_after(RateLimited(message="429 Please try again in 750ms"), default=9) == 0.75
    assert _retry_after(RateLimited(message="429 Please try again in 1.2s."), default=9) == 1.2
    assert _retry_after(RateLimited("999"), default=9) == 30.0
    assert _retry_after(RateLimited(), default=4) == 4