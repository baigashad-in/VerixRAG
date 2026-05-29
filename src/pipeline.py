"""
The complete RAG pipeline: query -> answer.
This is the main entry point for application.
"""

from src.query.enhanced_retrieval import EnhancedRetriever, QueryStrategy
from src.generation.generator import RAGGenerator, GeneratedResponse
from src.generation.hallucination_check import HallucinationDetector


class RAGPipeline:
    """End-to-end RAG: transform -> retrieve -> generate -> verify."""

    def __init__(self, enhanced_retriever: EnhancedRetriever, generator: RAGGenerator, hallucination_detector: HallucinationDetector = None):
        self.retriever = enhanced_retriever
        self.generator = generator
        self.hallucination_detector = hallucination_detector

    def answer(self, query: str,  top_k: int = 5,
               query_strategy: QueryStrategy = QueryStrategy.NONE,
               check_hallucination: bool = True) -> dict:
        
        results = self.retriever.retrieve(
            query = query,
            top_k = top_k,
            strategy = query_strategy,
        )

        if not results:
            return {
                "answer": "I couldn't find any relevant documents "
                          "to answer this question.",
                "citations": [],
                "retrieval_results": [],
                "hallucination_check": None,
            }
        
        response = self.generator.generate(
            query = query,
            retrieval_results = results,
        )

        hallucination_check = None
        if check_hallucination and self.detector:
            hallucination_check = self.detector.check_citation_coverage(
                response.answer,
                num_sources = len(results),
            )

            if hallucination_check["is_suspicious"]:
                print(f"⚠ Potential hallucination detected: "
                      f"{hallucination_check['uncited_claims']}")
                
        return {
            "answer": response.answer,
            "citations": [
                {
                    "id": c.citation_id,
                    "source": c.source_file,
                    "preview": c.content_preview,
                }
                for c in response.citations
            ],
            "retrieval_results": len(results),
            "hallucination_check": hallucination_check,
            "query_strategy": query_strategy.value,
            "model": response.model,
        }