"""
Chunk deduplication at ingest time.

Duplicated text wastes retrieval slots. In this knowledge base,
policies.md repeats sections of refund_policy.md, and a manual can
repeat the same heading block. With top_k=8, two copies of one
paragraph means one fewer distinct piece of evidence for the LLM.

TWO CHECKS:
1. Exact: identical text after normalising case, whitespace and
   markdown symbols -> duplicate.
2. Near: embedding cosine >= DEDUP_COSINE (0.95) AND word-set
   Jaccard >= DEDUP_JACCARD (0.85) -> duplicate.

Why both for "near"? Embeddings alone would merge "Business plan:
5000 requests/min" with "Standard plan: 1000 requests/min" — similar
meaning, different facts. Requiring heavy word overlap as well keeps
chunks whose numbers or names differ.

Within a batch, the most complete copy wins: chunks are considered
longest-first (most distinct words), ties broken by file name and
position, so results are deterministic. A truncated or templated copy
(e.g. a summary file missing a price) loses to the full one.
Chunks already stored from files that are not being re-ingested
always win over new ones, since they can't be swapped mid-batch.
"""

import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

_MD_NOISE = re.compile(r"[#*_`>|\-.,;:!?()\[\]\"']+")  # markdown + punctuation
_WS = re.compile(r"\s+")
_WORD = re.compile(r"[a-z0-9$%@]+")


def normalize(text: str) -> str:
    text = text.lower().replace("\ufeff", "")
    text = _MD_NOISE.sub(" ", text)
    return _WS.sub(" ", text).strip()


def fingerprint(text: str) -> str:
    import hashlib
    return hashlib.sha1(normalize(text).encode("utf-8")).hexdigest()[:16]


def word_set(text: str) -> set[str]:
    return set(_WORD.findall(normalize(text)))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


@dataclass
class DedupReport:
    kept: list = field(default_factory=list)
    dropped: list[dict] = field(default_factory=list)

    def dropped_by_source(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for d in self.dropped:
            counts[d["source"]] = counts.get(d["source"], 0) + 1
        return counts


class Deduplicator:
    def __init__(self, cosine_threshold: float = 0.95,
                 jaccard_threshold: float = 0.85):
        self.cosine_threshold = cosine_threshold
        self.jaccard_threshold = jaccard_threshold

    @staticmethod
    def _ref(chunk) -> str:
        meta = chunk.metadata
        return f"{meta.get('source', '?')}#chunk{meta.get('chunk_index', '?')}"

    def filter(self, chunks: list,
               external_lookup: Callable[[list[float]], list[dict]] | None = None
               ) -> DedupReport:
        """Split embedded chunks into kept and dropped.

        Args:
            chunks: EmbeddedChunk-like objects (content, embedding, metadata)
            external_lookup: optional fn(embedding) -> nearest stored chunks
                [{"content", "similarity", "source", "id"}], used to check
                against chunks already in the database.
        """
        report = DedupReport()
        kept_fps: dict[str, str] = {}
        kept_vecs: list[np.ndarray] = []
        kept_words: list[set[str]] = []
        kept_refs: list[str] = []

        # Most complete copy first, deterministic tie-break
        order = sorted(range(len(chunks)), key=lambda i: (
            -len(word_set(chunks[i].content)),
            str(chunks[i].metadata.get("source", "")),
            int(chunks[i].metadata.get("chunk_index", 0) or 0),
        ))
        kept_ids: set[int] = set()

        for i in order:
            chunk = chunks[i]
            ref = self._ref(chunk)
            fp = fingerprint(chunk.content)
            words = word_set(chunk.content)
            vec = np.asarray(chunk.embedding, dtype=np.float32)
            norm = np.linalg.norm(vec)
            vec = vec / norm if norm > 0 else vec

            duplicate = None
            if fp in kept_fps:
                duplicate = {"duplicate_of": kept_fps[fp], "reason": "exact", "similarity": 1.0}

            if duplicate is None and kept_vecs:
                sims = np.stack(kept_vecs) @ vec
                for idx in np.argsort(-sims):
                    if sims[idx] < self.cosine_threshold:
                        break
                    jac = jaccard(words, kept_words[idx])
                    if jac >= self.jaccard_threshold:
                        duplicate = {"duplicate_of": kept_refs[idx], "reason": "near",
                                     "similarity": round(float(sims[idx]), 4),
                                     "jaccard": round(jac, 3)}
                        break

            if duplicate is None and external_lookup is not None:
                for nb in external_lookup(chunk.embedding):
                    nb_ref = f"{nb.get('source', '?')}#id{nb.get('id', '?')}"
                    if fingerprint(nb["content"]) == fp:
                        duplicate = {"duplicate_of": nb_ref, "reason": "exact_stored",
                                     "similarity": 1.0}
                        break
                    if nb["similarity"] >= self.cosine_threshold:
                        jac = jaccard(words, word_set(nb["content"]))
                        if jac >= self.jaccard_threshold:
                            duplicate = {"duplicate_of": nb_ref, "reason": "near_stored",
                                         "similarity": round(float(nb["similarity"]), 4),
                                         "jaccard": round(jac, 3)}
                            break

            if duplicate:
                report.dropped.append({"source": chunk.metadata.get("source"),
                                       "chunk": ref, "preview": chunk.content[:80],
                                       **duplicate})
                continue

            chunk.metadata["chunk_fp"] = fp
            kept_ids.add(i)
            kept_fps[fp] = ref
            kept_vecs.append(vec)
            kept_words.append(words)
            kept_refs.append(ref)

        # Return kept chunks in their original (document) order
        report.kept = [chunks[i] for i in range(len(chunks)) if i in kept_ids]
        return report