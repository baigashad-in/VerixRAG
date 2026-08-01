"""
Request tracing and performance measurement (time + memory).

Detecting When an answer is wrong or slow.
E.g. "Re-ranking took 410 ms of a 2.3 s request, and the LLM call
used 1,840 prompt tokens" is slow. Every request is traced:

    trace_id  -> one per /api/chat call (or per eval question)
    spans     -> one per pipeline stage: classify, embed, hybrid
                 search, rerank, scoping, gate, LLM, citations...
    per span  -> wall time (ms) and process memory (RSS) before/after
    per trace -> total time, RSS delta, process peak RSS, tokens,
                 retrieval scores, and the final decision

HOW IT WORKS:
A `contextvars.ContextVar` holds the active trace, so any code
can call `span("retrieval.rerank")` without passing a trace object
around. If no trace is active, `span()` is a cheap no-op — library
code stays usable from scripts and tests.

MEMORY NUMBERS:
- rss_mb: resident memory of the whole process right now (Linux
  /proc/self/statm). Includes models, creating a baseline.
- rss delta per span: how much the process grew during the span.
  Python and the allocator reuse freed memory, so deltas are
  usually 0 after warm-up; a large delta flags a real allocation.
- peak_rss_mb: the process high-water mark since start
  (getrusage ru_maxrss). Use it to size the container.
- py_heap_peak_mb (optional, TRACE_TRACEMALLOC=1): peak Python
  heap allocated during this trace. Precise but adds ~10-30%
  overhead, default = off.
"""

from __future__ import annotations

import contextvars
import json
import os
import resource
import sys
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


# ── Memory helpers ─────────────────────────────────────────────

def _statm_mb() -> float | None:
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return round(pages * _PAGE_SIZE / 1_048_576, 1)
    except (OSError, ValueError, IndexError):
        return None


def _maxrss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KB, macOS reports bytes
    divisor = 1_048_576 if sys.platform == "darwin" else 1024
    return round(peak / divisor, 1)


def rss_mb() -> float:
    """Current resident set size of this process, in MB."""
    current = _statm_mb()
    return current if current is not None else _maxrss_mb()   # non-Linux fallback


def peak_rss_mb() -> float:
    """Peak resident set size of this process since start, in MB.

    The kernel updates the high-water mark lazily, so it can trail the
    current RSS by a few hundred KB; never report a peak below "now".
    """
    current = _statm_mb()
    return max(_maxrss_mb(), current or 0.0)


def _percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile (no numpy needed)."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, int(round(pct / 100 * len(ordered) + 0.4999)))
    return round(ordered[min(rank, len(ordered)) - 1], 1)


def latency_stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(sum(values) / len(values), 1),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "max": round(max(values), 1),
    }


# ── Trace model ────────────────────────────────────────────────

class Trace:
    """One end-to-end request: spans, memory, tokens and decisions."""

    def __init__(self, kind: str = "request", tracemalloc_enabled: bool = False,
                 **attrs):
        self.trace_id = uuid.uuid4().hex[:16]
        self.kind = kind
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.attrs: dict = dict(attrs)
        self.spans: list[dict] = []
        self.tokens = {"prompt": 0, "completion": 0, "total": 0}
        self.status = "running"
        self.total_ms: float | None = None
        self._t0 = time.perf_counter()
        self._stack: list[str] = []
        self.rss_start_mb = rss_mb()
        self.rss_end_mb: float | None = None
        self.peak_rss_mb: float | None = None
        self.py_heap_peak_mb: float | None = None
        self._tracemalloc = tracemalloc_enabled
        if self._tracemalloc:
            import tracemalloc
            if not tracemalloc.is_tracing():
                tracemalloc.start()
            tracemalloc.reset_peak()

    # -- recording --

    @contextmanager
    def span(self, name: str, **attrs):
        record = {
            "name": name,
            "parent": self._stack[-1] if self._stack else None,
            "offset_ms": round((time.perf_counter() - self._t0) * 1000, 1),
            "attrs": dict(attrs),
        }
        rss_before = rss_mb()
        start = time.perf_counter()
        self._stack.append(name)
        try:
            yield record["attrs"]          # callers may add attrs while running
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            raise
        finally:
            self._stack.pop()
            record["duration_ms"] = round((time.perf_counter() - start) * 1000, 2)
            rss_after = rss_mb()
            record["rss_mb"] = rss_after
            record["rss_delta_mb"] = round(rss_after - rss_before, 1)
            self.spans.append(record)

    def set(self, **attrs):
        self.attrs.update(attrs)

    def add_tokens(self, prompt: int = 0, completion: int = 0):
        self.tokens["prompt"] += int(prompt or 0)
        self.tokens["completion"] += int(completion or 0)
        self.tokens["total"] = self.tokens["prompt"] + self.tokens["completion"]

    def finish(self, status: str = "ok"):
        if self.total_ms is not None:      # already finished
            return
        self.status = status
        self.total_ms = round((time.perf_counter() - self._t0) * 1000, 1)
        self.rss_end_mb = rss_mb()
        self.peak_rss_mb = peak_rss_mb()
        if self._tracemalloc:
            import tracemalloc
            _, peak = tracemalloc.get_traced_memory()
            self.py_heap_peak_mb = round(peak / 1_048_576, 2)

    # -- views --

    def stage_ms(self) -> dict:
        """Total time per top-level stage name (children folded into parents)."""
        stages: dict[str, float] = {}
        for s in self.spans:
            if s["parent"] is None:
                stages[s["name"]] = round(stages.get(s["name"], 0.0) + s["duration_ms"], 1)
        return stages

    def summary(self) -> dict:
        """Compact view returned to API clients."""
        return {
            "trace_id": self.trace_id,
            "total_ms": self.total_ms,
            "stages_ms": self.stage_ms(),
            "memory_mb": {
                "rss_start": self.rss_start_mb,
                "rss_end": self.rss_end_mb,
                "rss_delta": (round(self.rss_end_mb - self.rss_start_mb, 1)
                              if self.rss_end_mb is not None else None),
                "process_peak": self.peak_rss_mb,
                **({"py_heap_peak": self.py_heap_peak_mb}
                   if self.py_heap_peak_mb is not None else {}),
            },
            "tokens": self.tokens,
        }

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "kind": self.kind,
            "started_at": self.started_at,
            "status": self.status,
            **self.summary(),
            "attrs": self.attrs,
            "spans": self.spans,
        }


# ── Trace store: in-memory ring buffer + optional JSONL sink ──

class TraceStore:
    """Keeps the last N traces for /api/traces and /api/metrics.

    The JSONL file (default experiments/traces.jsonl, which is a
    mounted volume) keeps history across restarts for offline analysis.
    """

    def __init__(self, maxlen: int = 500, log_path: str = "",
                 max_log_bytes: int = 50 * 1024 * 1024):
        self.max_log_bytes = max_log_bytes
        self._traces: deque[dict] = deque(maxlen=maxlen)
        self._index: dict[str, dict] = {}
        self._lock = threading.Lock()
        self.log_path = log_path
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.counters: dict[str, int] = {}

    def add(self, trace: Trace):
        record = trace.to_dict()
        with self._lock:
            if len(self._traces) == self._traces.maxlen:
                oldest = self._traces[0]
                self._index.pop(oldest["trace_id"], None)
            self._traces.append(record)
            self._index[record["trace_id"]] = record
        if self.log_path:
            try:
                path = Path(self.log_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists() and path.stat().st_size > self.max_log_bytes:
                    path.replace(path.with_suffix(path.suffix + ".1"))   # keep one rotation
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(record, default=str) + "\n")
            except OSError as exc:      # never fail a request over logging
                print(f"  trace log write failed: {exc}", flush=True)

    def incr(self, name: str, by: int = 1):
        with self._lock:
            self.counters[name] = self.counters.get(name, 0) + by

    def get(self, trace_id: str) -> dict | None:
        with self._lock:
            return self._index.get(trace_id)

    def recent(self, limit: int = 20, kind: str | None = None) -> list[dict]:
        with self._lock:
            items = [t for t in self._traces if kind is None or t["kind"] == kind]
        items = items[-limit:][::-1]
        return [
            {k: t[k] for k in ("trace_id", "kind", "started_at", "status", "total_ms")}
            | {"decision": t["attrs"].get("decision"),
               "query": t["attrs"].get("query"),
               "tokens": t["tokens"]["total"]}
            for t in items
        ]

    def clear(self):
        with self._lock:
            self._traces.clear()
            self._index.clear()
            self.counters.clear()

    def metrics(self, kind: str | None = None) -> dict:
        """Aggregate latency, memory, tokens and decisions over buffered traces."""
        with self._lock:
            traces = [t for t in self._traces if kind is None or t["kind"] == kind]
            counters = dict(self.counters)

        totals = [t["total_ms"] for t in traces if t.get("total_ms") is not None]
        per_stage: dict[str, list[float]] = {}
        for t in traces:
            for name, ms in t.get("stages_ms", {}).items():
                per_stage.setdefault(name, []).append(ms)

        decisions: dict[str, int] = {}
        for t in traces:
            d = t["attrs"].get("decision", "unknown")
            decisions[d] = decisions.get(d, 0) + 1

        tokens = [t["tokens"]["total"] for t in traces if t["tokens"]["total"]]
        rss_deltas = [t["memory_mb"]["rss_delta"] for t in traces
                      if t["memory_mb"].get("rss_delta") is not None]

        return {
            "window": {"traces": len(traces), "since": self.started_at},
            "latency_ms": latency_stats(totals),
            "stage_latency_ms": {k: latency_stats(v) for k, v in sorted(per_stage.items())},
            "decisions": decisions,
            "tokens_per_request": latency_stats(tokens),
            "memory_mb": {
                "rss_now": rss_mb(),
                "process_peak": peak_rss_mb(),
                "rss_delta_per_request": latency_stats(rss_deltas),
            },
            "counters": counters,
        }


_store: TraceStore | None = None


def get_trace_store() -> TraceStore:
    global _store
    if _store is None:
        from src.config import get_settings
        s = get_settings()
        _store = TraceStore(maxlen=s.trace_buffer_size, log_path=s.trace_log_path)
    return _store


def set_trace_store(store: TraceStore):
    """Swap the global store (tests)."""
    global _store
    _store = store


# ── Context helpers ────────────────────────────────────────────

_current: contextvars.ContextVar[Trace | None] = contextvars.ContextVar(
    "verix_trace", default=None)


def current_trace() -> Trace | None:
    return _current.get()


@contextmanager
def start_trace(kind: str = "request", store: TraceStore | None = None, **attrs):
    """Open a new trace, make it current, finish and store it on exit."""
    from src.config import get_settings
    trace = Trace(kind=kind, tracemalloc_enabled=get_settings().trace_tracemalloc, **attrs)
    token = _current.set(trace)
    status = "ok"
    try:
        yield trace
    except Exception:
        status = "error"
        raise
    finally:
        _current.reset(token)
        trace.finish(status if trace.status == "running" else trace.status)
        (store or get_trace_store()).add(trace)


@contextmanager
def ensure_trace(kind: str = "pipeline", **attrs):
    """Reuse the active trace if there is one, otherwise open a new one.

    The API opens a trace per request; the eval runner and scripts
    call the pipeline directly. ensure_trace makes both get traces
    without double-counting.
    """
    active = current_trace()
    if active is not None:
        active.set(**attrs)
        yield active
        return
    with start_trace(kind=kind, **attrs) as trace:
        yield trace


@contextmanager
def span(name: str, **attrs):
    """Time a stage inside the active trace; no-op when none is active."""
    trace = current_trace()
    if trace is None:
        yield {}
        return
    with trace.span(name, **attrs) as span_attrs:
        yield span_attrs


@contextmanager
def measure(name: str, sink: dict | None = None):
    """Standalone timer + memory probe for startup and ingestion.

    Usage:
        perf = {}
        with measure("load.embedder", perf):
            embedder = EmbeddingService(...)
        perf -> {"load.embedder": {"ms": 812.4, "rss_delta_mb": 96.1, ...}}
    """
    rss_before = rss_mb()
    start = time.perf_counter()
    result: dict = {}
    try:
        yield result
    finally:
        result.update({
            "ms": round((time.perf_counter() - start) * 1000, 1),
            "rss_before_mb": rss_before,
            "rss_after_mb": rss_mb(),
            "rss_delta_mb": round(rss_mb() - rss_before, 1),
        })
        if sink is not None:
            sink[name] = dict(result)