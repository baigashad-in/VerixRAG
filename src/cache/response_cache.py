"""
Full-response cache for /api/chat.

A hit skips the guardrail classifier and the LLM entirely. The cached
response is returned with "cache": {"hit": true, "age_s": ...} so the
client (and the trace) can tell it apart from a fresh answer.
"""

from src.cache.ttl_cache import TTLCache, make_key, normalize_query


class ResponseCache:
    def __init__(self, store, pipeline_signature: str, max_entries: int = 512,
                 ttl_seconds: float = 3600, enabled: bool = True):
        self.store = store                      # anything with index_version()
        self.signature = pipeline_signature
        self.enabled = enabled
        self.cache = TTLCache(max_entries, ttl_seconds, name="response")

    def key(self, query: str, top_k: int, strategy: str) -> str:
        version = self.store.index_version() if hasattr(self.store, "index_version") else "na"
        return make_key("response", normalize_query(query), top_k, strategy,
                        version, self.signature)

    def get(self, query: str, top_k: int, strategy: str):
        if not self.enabled:
            return False, None, None
        return self.cache.get(self.key(query, top_k, strategy))

    @staticmethod
    def cacheable(result: dict) -> bool:
        guard = result.get("guardrails") or {}
        if guard.get("input_pii_detected"):          # never keep answers to PII queries
            return False
        if guard.get("original_blocked"):            # safety-filtered output
            return False
        return bool(result.get("answer"))

    def put(self, query: str, top_k: int, strategy: str, result: dict) -> bool:
        if not self.enabled or not self.cacheable(result):
            return False
        stored = {k: v for k, v in result.items() if k not in ("perf", "trace_id", "cache")}
        stored["_original_trace_id"] = result.get("trace_id")
        self.cache.set(self.key(query, top_k, strategy), stored)
        return True

    def cache_stats(self) -> dict:
        return {**self.cache.stats(), "enabled": self.enabled}

    def on_index_changed(self):
        # Keys include the index version, so old entries are already
        # unreachable; clearing just frees the memory immediately.
        self.cache.clear()