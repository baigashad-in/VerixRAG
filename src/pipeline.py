"""
The complete RAG pipeline: query -> answer.
Main entry point for application.
"""

from src.query.enhanced_retrieval import EnhancedRetriever, QueryStrategy
from src.generation.generator import RAGGenerator
from src.generation.hallucination_check import HallucinationDetector
from src.observability.tracing import ensure_trace, span
from src.retrieval.confidence import ABSTAIN_ANSWER, ConfidenceGate


class RAGPipeline:
    """End-to-end RAG: transform -> retrieve -> generate -> verify."""

    def __init__(self, enhanced_retriever: EnhancedRetriever, generator: RAGGenerator, hallucination_detector: HallucinationDetector = None,
                 confidence_gate: ConfidenceGate = None):
        self.retriever = enhanced_retriever
        self.generator = generator
        self.hallucination_detector = hallucination_detector
        if confidence_gate is None:
            from src.config import get_settings
            confidence_gate = ConfidenceGate(get_settings().rerank_min_score)
        self.gate = confidence_gate

    def answer(self, query: str,  top_k: int = 8,
               query_strategy: QueryStrategy = QueryStrategy.NONE,
               check_hallucination: bool = True,
               return_chunks: bool = False) -> dict:

        with ensure_trace("pipeline") as trace:
            result = self._answer(query, top_k, query_strategy,
                                  check_hallucination, return_chunks, trace)
            # A pipeline-owned trace finishes when this block exits; an
            # API-owned trace is still running and gets a fresh summary there.
        result["perf"] = trace.summary()
        return result

    def _answer(self, query, top_k, query_strategy, check_hallucination,
                return_chunks, trace) -> dict:
        trace.set(query=query[:200], top_k=top_k, strategy=query_strategy.value)

        with span("retrieval", strategy=query_strategy.value) as attrs:
            results = self.retriever.retrieve(
                query = query,
                top_k = top_k,
                strategy = query_strategy,
            )
            attrs["results"] = len(results)
        trace.set(retrieved=_describe(results))

        if not results:
            trace.set(decision="no_results")
            return {
                "answer": "I couldn't find any relevant documents "
                          "to answer this question.",
                "citations": [],
                "retrieval_results": [],
                "hallucination_check": None,
                "trace_id": trace.trace_id,
            }

        # Confidence gate: abstain without an LLM call when even the
        # best chunk is a poor match for the question.
        with span("confidence_gate") as attrs:
            gate = self.gate.check(results)
            attrs.update(gate.as_dict())
        trace.set(gate=gate.as_dict())
        if not gate.passed:
            trace.set(decision="abstained_low_confidence")
            return {
                "answer": ABSTAIN_ANSWER,
                "citations": [],
                "retrieval_results": len(results),
                **({"chunks": results} if return_chunks else {}),
                "hallucination_check": None,
                "abstained": gate.as_dict(),
                "query_strategy": query_strategy.value,
                "trace_id": trace.trace_id,
            }
        
        with span("generation"):
            response = self.generator.generate(
                query = query,
                retrieval_results = results,
            )

        hallucination_check = None
        if check_hallucination and self.hallucination_detector:
            with span("hallucination_check"):
                hallucination_check = self.hallucination_detector.check_citation_coverage(
                    response.answer,
                    num_sources = len(results),
                )

            if hallucination_check["is_suspicious"]:
                print(f"⚠ Potential hallucination detected: "
                      f"{hallucination_check['uncited_claims']}")

        trace.set(decision="answered", model=response.model)
        return {
            "answer": response.answer,
            "citations": [
                {
                    "id": c.citation_id,
                    "source": c.source_file,
                    "section": c.section,
                    "preview": c.content_preview,
                }
                for c in response.citations
            ],
            "retrieval_results": len(results),
            **({"chunks": results} if return_chunks else {}),
            "hallucination_check": hallucination_check,
            "confidence": gate.as_dict(),
            "query_strategy": query_strategy.value,
            "model": response.model,
            "usage": response.usage,
            "trace_id": trace.trace_id,
        }


def _describe(results: list) -> list[dict]:
    """Compact, JSON-safe view of retrieved chunks for the trace."""
    described = []
    for rank, r in enumerate(results, 1):
        meta = getattr(r, "metadata", {}) or {}
        item = {
            "rank": rank,
            "chunk_id": getattr(r, "chunk_id", None),
            "source": meta.get("source") or meta.get("filename"),
        }
        for field_name in ("rerank_score", "final_score", "rrf_score"):
            value = getattr(r, field_name, None)
            if value is not None:
                item[field_name] = round(float(value), 4)
        for meta_key in ("domain", "section"):
            if meta.get(meta_key):
                item[meta_key] = meta[meta_key]
        described.append(item)
    return described