"""
Unified guardrails pipeline: wraps the RAG pipeline with
input/output protection.

API endpoint actually calls this directly - not the
RAG pipeline.
"""

from src.guardrails.query_classifier import QueryClassifierPipeline
from src.guardrails.pii_detector import PIIDetector
from src.guardrails.safety_filter import SafetyFilter
from src.pipeline import RAGPipeline
from src.query.enhanced_retrieval import QueryStrategy
from src.observability.tracing import current_trace, span

class GuardedRAGPipeline:
    """RAG pipeline wrapped with safety guardrails.
    
    FLOW:
    1. Classify query (in-scope?)
    2. Detect and redact PII from input
    3. Run RAG pipeline
    4. Check response safety
    5. Detect and redact PII from output
    6. Return safe, clean response
    """

    def __init__(self, rag_pipeline: RAGPipeline,
                 classifier: QueryClassifierPipeline = None,
                 pii_detector: PIIDetector = None,
                 safety_filter: SafetyFilter = None):
        self.rag = rag_pipeline
        self.classifier = classifier or QueryClassifierPipeline()
        self.pii = pii_detector or PIIDetector()
        self.safety = safety_filter or SafetyFilter()

    def answer(self, query: str, top_k: int = 5,
               query_strategy: QueryStrategy = QueryStrategy.NONE) -> dict:
        trace = current_trace()

        # -- STEP 1: Classify --
        with span("guardrails.classify") as attrs:
            classification = self.classifier.classify(query)
            attrs["stage"] = classification.get("stage")
            attrs["in_scope"] = classification.get("in_scope")

        if not classification["in_scope"]:
            if trace is not None:
                trace.set(decision="out_of_scope")
            return {
                "answer": "That question is outside what I can help with. "
                    "I can answer questions about our policies, "
                    "products, and account management.",
                "guardrails": {
                    "classified_as": "out_of_scope",
                    "reason": classification["reason"],
                    "stage": classification["stage"],
                },
            }
        
        # -- STEP 2: Redact PII from input --
        with span("guardrails.pii_input"):
            clean_query, input_pii = self.pii.redact(query)
        if trace is not None:
            # store the redacted query in traces, never the raw one
            trace.set(query=clean_query[:200], input_pii=len(input_pii))

        if input_pii:
            print(f"⚠ PII detected in query: "
                  f"{[m.pii_type for m in input_pii]}")
            

        # -- STEP 3: Run RAG pipeline with cleaned query --

        result = self.rag.answer(
            query = clean_query,
            top_k = top_k,
            query_strategy = query_strategy,
        )

        # -- STEP 4: Check response safety --
        with span("guardrails.safety"):
            safety_result = self.safety.check(result["answer"])
        if not safety_result.is_safe:
            if trace is not None:
                trace.set(decision="safety_blocked")
            result["answer"] = safety_result.filtered_response
            result["guardrails"] = {
                "safety_flags": safety_result.flags,
                "original_blocked": True,
            }
            return result
        
        # -- STEP 5: Redact PII from output --
        with span("guardrails.pii_output"):
            clean_answer, output_pii = self.pii.redact(result["answer"])
        result["answer"] = clean_answer

        if output_pii:
            print(f"⚠ PII detected in response: "
                  f"{[m.pii_type for m in output_pii]}")
            
        # -- STEP 6: Return clean result --
        result["guardrails"] = {
            "classified_as": "in_scope",
            "input_pii_detected": len(input_pii),
            "output_pii_detected": len(output_pii),
            "safety_flags": [],
        }

        return result