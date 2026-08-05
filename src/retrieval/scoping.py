"""
Source scoping: keep answers inside the domain the question is about.

THE BUG FIXED:
The knowledge base mixes several products and audiences: the store
(accounts, refunds, shipping), CloudDeck Pro (a separate product),
the VerixAPI developer docs, and the Nexora HR handbook. Asked for
"the support phone number", retrieval happily returned CloudDeck's
enterprise line. Faithfulness scored it 1.0 — the number IS in a
source — but it was the wrong product's number.

WORKING:
1. Every document is tagged with a domain at ingest
   (config/domains.json, or `domain:` front matter).
2. For each query we detect its domain:
   - alias match: the query names a product ("CloudDeck", "Nexora")
     -> that domain. Naming two domains = a cross-domain question,
     so no scoping.
   - centroid match: cosine between the query embedding and each
     domain's mean chunk embedding. Only used if the best domain
     beats the runner-up by DOMAIN_MIN_MARGIN.
3. Off-domain chunks get DOMAIN_PENALTY subtracted from their
   cross-encoder score, and the list is re-sorted.

WHY A SOFT PENALTY, NOT A FILTER:
Some questions legitimately span domains ("does the web password rule
match the API's?"). A strongly relevant off-domain chunk (score +8)
still beats a weak in-domain one (+3) after a 2.0 penalty; a marginal
off-domain chunk loses its slot to a relevant in-domain one.
The confidence gate always uses the raw score, so scoping can never
make the system refuse.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class DomainConfig:
    files: dict[str, str] = field(default_factory=dict)          # filename -> domain
    aliases: dict[str, list[str]] = field(default_factory=dict)  # domain -> names

    def domain_for(self, filename: str, front_matter: dict | None = None) -> str | None:
        if front_matter and front_matter.get("domain"):
            return str(front_matter["domain"]).strip().lower()
        domain = self.files.get(filename)
        return domain.strip().lower() if domain else None


def load_domain_config(path: str | None) -> DomainConfig:
    """Read the filename -> domain mapping. Missing file = no domains.

    Accepted formats:
      {"files": {"a.md": "store"}, "domains": {"store": {"aliases": [...]}}}
      {"sources": [{"filename": "a.md", "domain": "store"}]}   # documents_real/_manifest.json
      {"a.md": "store"}
    """
    if not path or not Path(path).is_file():
        return DomainConfig()
    raw = json.loads(Path(path).read_text())
    cfg = DomainConfig()
    if "files" in raw or "domains" in raw:
        cfg.files = {k: v for k, v in raw.get("files", {}).items() if isinstance(v, str)}
        for domain, spec in raw.get("domains", {}).items():
            aliases = spec.get("aliases", []) if isinstance(spec, dict) else spec
            cfg.aliases[domain.lower()] = [a for a in aliases if isinstance(a, str) and a.strip()]
    elif "sources" in raw:
        cfg.files = {s["filename"]: s["domain"] for s in raw["sources"]
                     if s.get("filename") and s.get("domain")}
    else:
        cfg.files = {k: v for k, v in raw.items() if isinstance(v, str)}
    return cfg


class DomainScorer:
    def __init__(self, aliases: dict[str, list[str]] | None = None,
                 penalty: float = 2.0, min_margin: float = 0.05,
                 enabled: bool = True):
        self.enabled = enabled
        self.penalty = penalty
        self.min_margin = min_margin
        self._alias_patterns = {
            domain: [re.compile(r"(?<![a-z0-9])" + re.escape(a.lower()) + r"(?![a-z0-9])")
                     for a in names]
            for domain, names in (aliases or {}).items()
        }
        self.centroids: dict[str, np.ndarray] = {}

    @classmethod
    def from_settings(cls, settings=None) -> "DomainScorer":
        from src.config import get_settings
        s = settings or get_settings()
        cfg = load_domain_config(s.domains_config)
        return cls(aliases=cfg.aliases, penalty=s.domain_penalty,
                   min_margin=s.domain_min_margin, enabled=s.scoping_enabled)

    def refresh(self, centroids: dict[str, list[float]]):
        """Load per-domain mean embeddings (call at startup and after ingest)."""
        normed = {}
        for domain, vec in centroids.items():
            v = np.asarray(vec, dtype=np.float32)
            n = np.linalg.norm(v)
            if n > 0:
                normed[domain] = v / n
        self.centroids = normed

    def detect(self, query: str, query_embedding: list[float] | None) -> dict:
        if not self.enabled:
            return {"domain": None, "method": "disabled"}

        q = query.lower()
        hits = sorted(d for d, pats in self._alias_patterns.items()
                      if any(p.search(q) for p in pats))
        if len(hits) == 1:
            return {"domain": hits[0], "method": "alias"}
        if len(hits) > 1:
            return {"domain": None, "method": "alias_multiple", "domains": hits}

        if len(self.centroids) < 2 or query_embedding is None:
            return {"domain": None, "method": "no_centroids"}
        qv = np.asarray(query_embedding, dtype=np.float32)
        n = np.linalg.norm(qv)
        if n == 0:
            return {"domain": None, "method": "no_centroids"}
        qv = qv / n
        scores = {d: float(c @ qv) for d, c in self.centroids.items()}
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        margin = ranked[0][1] - ranked[1][1]
        result = {"scores": {d: round(s, 4) for d, s in ranked},
                  "margin": round(margin, 4)}
        if margin >= self.min_margin:
            return {"domain": ranked[0][0], "method": "centroid", **result}
        return {"domain": None, "method": "centroid_uncertain", **result}

    def apply(self, results: list, detection: dict) -> list:
        """Penalise off-domain chunks and re-sort by the adjusted score."""
        target = detection.get("domain")
        for r in results:
            if getattr(r, "rerank_score", None) is None:
                return results           # nothing to adjust without cross-encoder scores
            r.final_score = r.rerank_score
            r.domain_penalty = 0.0
            chunk_domain = (r.metadata or {}).get("domain")
            if target and chunk_domain and chunk_domain != target:
                r.final_score = r.rerank_score - self.penalty
                r.domain_penalty = self.penalty
        results.sort(key=lambda r: r.final_score, reverse=True)
        return results