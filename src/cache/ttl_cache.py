"""
A small thread-safe LRU cache with time-to-live, plus key helpers.

WHY TWO CACHES (see RetrievalEngine and ResponseCache):
- Retrieval cache: query -> retrieved chunks. Skips embedding,
  hybrid search and cross-encoder re-ranking (CPU-heavy) on repeats.
- Response cache: query -> full answer. Skips the guardrail
  classifier and the LLM call, which is what actually costs time
  and free-tier quota.

CORRECTNESS RULES:
1. Every key includes the INDEX VERSION, so re-ingesting documents
   makes every old entry unreachable — no stale answers.
2. The response key also includes a PIPELINE SIGNATURE (model,
   system prompt, gate threshold, scoping settings), so changing the
   prompt or a threshold never serves answers made under the old one.
3. Queries are normalised (case, whitespace, trailing punctuation)
   so "What is X?" and "what is x" share an entry.
4. Answers to queries containing PII are never cached.
"""

import copy
import hashlib
import json
import re
import threading
import time
from collections import OrderedDict

_WS = re.compile(r"\s+")


def normalize_query(query: str) -> str:
    q = _WS.sub(" ", query.strip().lower())
    return q.rstrip(" ?!.")


def make_key(*parts) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class TTLCache:
    def __init__(self, max_entries: int = 512, ttl_seconds: float = 3600,
                 name: str = "cache"):
        self.max_entries = max_entries
        self.ttl = ttl_seconds
        self.name = name
        self._data: OrderedDict[str, tuple[float, object]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = self.evictions = self.expired = 0

    def get(self, key: str):
        """Return (hit, value, age_seconds)."""
        now = time.monotonic()
        with self._lock:
            item = self._data.get(key)
            if item is None:
                self.misses += 1
                return False, None, None
            stored_at, value = item
            if now - stored_at > self.ttl:
                del self._data[key]
                self.expired += 1
                self.misses += 1
                return False, None, None
            self._data.move_to_end(key)
            self.hits += 1
            return True, copy.deepcopy(value), round(now - stored_at, 1)

    def set(self, key: str, value):
        with self._lock:
            self._data[key] = (time.monotonic(), copy.deepcopy(value))
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)
                self.evictions += 1

    def clear(self):
        with self._lock:
            self._data.clear()

    def stats(self) -> dict:
        with self._lock:
            lookups = self.hits + self.misses
            return {
                "name": self.name,
                "entries": len(self._data),
                "max_entries": self.max_entries,
                "ttl_seconds": self.ttl,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / lookups, 3) if lookups else None,
                "evictions": self.evictions,
                "expired": self.expired,
            }