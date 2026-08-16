"""Tests for request tracing and performance measurement."""

import time

import pytest

from src.observability.tracing import (
    TraceStore, current_trace, ensure_trace, latency_stats, measure,
    peak_rss_mb, rss_mb, set_trace_store, span, start_trace,
)


@pytest.fixture
def store():
    s = TraceStore(maxlen=10, log_path="")
    set_trace_store(s)
    yield s
    set_trace_store(TraceStore(maxlen=10, log_path=""))


def test_memory_probes_return_positive_numbers():
    assert rss_mb() > 0
    assert peak_rss_mb() >= rss_mb() * 0.5   # peak is a lifetime high-water mark


def test_span_without_trace_is_noop():
    assert current_trace() is None
    with span("anything") as attrs:
        attrs["x"] = 1          # must not raise


def test_trace_records_spans_time_and_memory(store):
    with start_trace("chat", store=store) as trace:
        with span("retrieval") as attrs:
            with span("retrieval.rerank"):
                time.sleep(0.01)
            attrs["results"] = 3
        trace.add_tokens(prompt=100, completion=20)
        trace.set(decision="answered")

    saved = store.get(trace.trace_id)
    assert saved["status"] == "ok"
    assert saved["total_ms"] >= 10
    names = [s["name"] for s in saved["spans"]]
    assert names == ["retrieval.rerank", "retrieval"]        # children finish first
    rerank = saved["spans"][0]
    assert rerank["parent"] == "retrieval" and rerank["duration_ms"] >= 10
    assert "rss_mb" in rerank and "rss_delta_mb" in rerank
    # top-level stage totals only count parents, so time isn't double-counted
    assert set(saved["stages_ms"]) == {"retrieval"}
    assert saved["tokens"] == {"prompt": 100, "completion": 20, "total": 120}
    assert saved["memory_mb"]["process_peak"] > 0


def test_error_marks_trace_and_span(store):
    with pytest.raises(ValueError):
        with start_trace("chat", store=store) as trace:
            with span("generation.llm"):
                raise ValueError("boom")
    saved = store.get(trace.trace_id)
    assert saved["status"] == "error"
    assert "boom" in saved["spans"][0]["error"]


def test_ensure_trace_reuses_active_trace(store):
    with start_trace("chat", store=store) as outer:
        with ensure_trace("pipeline") as inner:
            assert inner is outer
    assert len(store.recent(10)) == 1                 # stored once, by the owner


def test_ensure_trace_creates_trace_when_none_active(store):
    with ensure_trace("pipeline") as trace:
        pass
    assert store.get(trace.trace_id)["kind"] == "pipeline"


def test_metrics_aggregate_latency_and_decisions(store):
    for decision in ("answered", "answered", "abstained"):
        with start_trace("chat", store=store) as t:
            with span("retrieval"):
                pass
            t.set(decision=decision)
    m = store.metrics()
    assert m["window"]["traces"] == 3
    assert m["decisions"] == {"answered": 2, "abstained": 1}
    assert m["latency_ms"]["n"] == 3 and m["latency_ms"]["p95"] is not None
    assert "retrieval" in m["stage_latency_ms"]


def test_ring_buffer_evicts_oldest(store):
    ids = []
    for _ in range(12):
        with start_trace("chat", store=store) as t:
            pass
        ids.append(t.trace_id)
    assert store.get(ids[0]) is None and store.get(ids[-1]) is not None
    assert store.metrics()["window"]["traces"] == 10


def test_jsonl_sink(tmp_path):
    log = tmp_path / "traces.jsonl"
    s = TraceStore(maxlen=5, log_path=str(log))
    with start_trace("chat", store=s):
        pass
    assert log.read_text().count("\n") == 1


def test_latency_stats_percentiles():
    stats = latency_stats([float(i) for i in range(1, 101)])
    assert stats["p50"] == 50.0 and stats["p95"] == 95.0 and stats["max"] == 100.0
    assert latency_stats([]) == {"n": 0}


def test_measure_records_time_and_memory():
    sink = {}
    with measure("load", sink) as m:
        m["items"] = 3
        time.sleep(0.005)
    assert sink["load"]["ms"] >= 5 and sink["load"]["items"] == 3
    assert "rss_delta_mb" in sink["load"]


def test_jsonl_sink_rotates(tmp_path):
    log = tmp_path / "traces.jsonl"
    s = TraceStore(maxlen=5, log_path=str(log), max_log_bytes=10)
    for _ in range(3):
        with start_trace("chat", store=s):
            pass
    assert (tmp_path / "traces.jsonl.1").exists()
    assert log.read_text().count("\n") == 1
