"""
Central runtime settings:

Behaviour (confidence gate, domain scoping, dedup,caching, 
tracing) can be switched off by setting its value to an empty 
string or 0.

Settings are read when `get_settings()` is first called and then
cached, so tests can call `reset_settings()` after changing os.environ.
"""

from dataclasses import dataclass, asdict
import os


def _float_or_none(name: str, default: str) -> float | None:
    raw = os.getenv(name, default).strip()
    if raw == "" or raw.lower() in ("none", "off", "disabled"):
        return None
    return float(raw)


def _int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else default


def _float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else default


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    # -- Confidence gate --
    # Cross-encoder logit below which we abstain without calling the LLM.
    # None disables the gate. Calibrate with scripts/calibrate.py.
    rerank_min_score: float | None

    # -- Source scoping --
    scoping_enabled: bool
    domain_penalty: float          # subtracted from off-domain rerank scores
    domain_min_margin: float       # centroid similarity margin needed to pick a domain
    domains_config: str            # path to the filename -> domain mapping
    
    # -- Hybrid retrieval --
    hybrid_sparse_weight: float        # BM25 weight in RRF fusion (dense weight is 1.0)

    # -- Keyword scope classifier --
    keyword_scope_enabled: bool        # seed in-scope terms from the documents
    keyword_scope_body_terms: int      # distinctive body words per document
    keyword_scope_min_matches: int     # terms a query must match to skip the LLM

    # -- Deduplication --
    dedup_enabled: bool
    dedup_cosine: float            # near-duplicate if cosine >= this ...
    dedup_jaccard: float           # ... AND token Jaccard >= this

    # -- Caching --
    cache_enabled: bool
    cache_ttl_seconds: int
    cache_max_entries: int

    # -- Observability --
    trace_buffer_size: int         # traces kept in memory for /api/traces
    trace_log_path: str            # JSONL sink ('' disables)
    trace_tracemalloc: bool        # per-request Python heap peak (adds overhead)

    def as_dict(self) -> dict:
        return asdict(self)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings(
            rerank_min_score=_float_or_none("RERANK_MIN_SCORE", "-6.0"),
            scoping_enabled=_bool("SCOPING_ENABLED", True),
            domain_penalty=_float("DOMAIN_PENALTY", 2.0),
            domain_min_margin=_float("DOMAIN_MIN_MARGIN", 0.05),
            domains_config=os.getenv("DOMAINS_CONFIG", "config/domains.json"),
            hybrid_sparse_weight=_float("HYBRID_SPARSE_WEIGHT", 1.0),
            keyword_scope_enabled=_bool("KEYWORD_SCOPE_ENABLED", True),
            keyword_scope_body_terms=_int("KEYWORD_SCOPE_BODY_TERMS", 10),
            keyword_scope_min_matches=_int("KEYWORD_SCOPE_MIN_MATCHES", 2),
            dedup_enabled=_bool("DEDUP_ENABLED", True),
            dedup_cosine=_float("DEDUP_COSINE", 0.95),
            dedup_jaccard=_float("DEDUP_JACCARD", 0.85),
            cache_enabled=_bool("CACHE_ENABLED", True),
            cache_ttl_seconds=_int("CACHE_TTL_SECONDS", 3600),
            cache_max_entries=_int("CACHE_MAX_ENTRIES", 512),
            trace_buffer_size=_int("TRACE_BUFFER_SIZE", 500),
            trace_log_path=os.getenv("TRACE_LOG_PATH", "experiments/traces.jsonl"),
            trace_tracemalloc=_bool("TRACE_TRACEMALLOC", False),
        )
    return _settings


def reset_settings():
    """Forget cached settings (tests change os.environ, then call this)."""
    global _settings
    _settings = None