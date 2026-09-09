"""Load test: question selection, phases, decisions and summary."""

import json
import urllib.error

import pytest

from scripts import load_test


def write_eval(tmp_path):
    items = [{"question": f"in {i}", "category": "easy"} for i in range(6)]
    items += [{"question": "weather?", "category": "out_of_scope"}, {"question": "in 0"}]
    path = tmp_path / "eval.json"
    path.write_text(json.dumps({"questions": items}))
    return str(path)


def test_pick_questions_in_scope_unique_and_seeded(tmp_path):
    path = write_eval(tmp_path)
    assert load_test.pick_questions(path, None, None) == [f"in {i}" for i in range(6)]
    a, b = load_test.pick_questions(path, 4, 7), load_test.pick_questions(path, 4, 7)
    assert a == b and len(a) == 4 and "weather?" not in a
    assert sorted(load_test.pick_questions(path, None, 7)) == sorted(f"in {i}" for i in range(6))


def test_decision():
    assert load_test.decision({"cache": {"hit": True}}) == "cache_hit"
    assert load_test.decision({"guardrails": {"classified_as": "out_of_scope"}}) == "out_of_scope"
    assert load_test.decision({"abstained": {"passed": False}}) == "abstained"
    assert load_test.decision({"answer": "x", "cache": {"hit": False}}) == "answered"


def fake_api(store):
    """Stand-in for /api/chat with a response cache."""
    calls = []

    def call(base, api_key, body):
        calls.append(body)
        q = body["query"]
        if q == "boom":
            raise urllib.error.HTTPError(base, 429, "busy", {}, None)
        hit = body["use_cache"] and q in store
        if body["use_cache"]:
            store.add(q)
        return {"answer": "a", "trace_id": "t", "cache": {"hit": hit},
                "perf": {"total_ms": 1.0 if hit else 900.0, "tokens": {"total": 0 if hit else 1500}}}
    return call, calls


def test_phases_store_then_hit_and_summary(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("API_KEY", "test")
    call, calls = fake_api(set())
    monkeypatch.setattr(load_test, "_call", call)
    out = tmp_path / "out"
    code = load_test.main(["--dataset", write_eval(tmp_path), "--limit", "5", "--cached", "2",
                           "--seed", "3", "--output-dir", str(out)])
    assert code == 0
    report = json.loads(next(out.glob("load_test_default_*.json")).read_text())
    fresh, cached = report["fresh"], report["cached"]
    assert [c["use_cache"] for c in calls[:5]] == [True, True, False, False, False]
    assert fresh["requests"] == 5 and fresh["decisions"] == {"answered": 5}
    assert fresh["server_ms"]["p50"] == 900.0 and fresh["tokens_per_answer"] == 1500
    assert cached["decisions"] == {"cache_hit": 2} and cached["server_ms"]["p50"] == 1.0
    assert cached["unexpected"] == 0
    assert f"trace_report --since {report['started_at']}" in capsys.readouterr().out


def test_stale_cache_hit_is_counted_not_timed_and_errors_fail(monkeypatch, tmp_path):
    monkeypatch.setenv("API_KEY", "test")
    path = write_eval(tmp_path)
    first = load_test.pick_questions(path, 3, 7)[0]
    call, _ = fake_api({first})                       # left in the cache by an earlier run
    rows = load_test.run_phase("http://x", "k", [first, "boom"], True, 0, 1, call)
    s = load_test.summarize(rows, "fresh")
    assert s["errors"] == 1 and s["unexpected"] == 1 and s["client_ms"] == {"n": 0}


def test_concurrency_keeps_order(monkeypatch):
    call, _ = fake_api(set())
    rows = load_test.run_phase("http://x", "k", ["a", "b", "c"], False, 0, 3, call)
    assert [r["query"] for r in rows] == ["a", "b", "c"]


def test_missing_api_key_is_an_error(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    with pytest.raises(SystemExit):
        load_test.main(["--limit", "1"])
