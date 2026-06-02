"""
CLI script to A/B test different RAG configurations.

Usage:
    python scripts/compare_strategies.py

This runs evaluation with multiple chunking strategies
and produces a comparison report.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.ingestion.ingest import IngestionPipeline
from src.storage.vector_store import VectorStore
from src.retrieval.sparse_retrieval import BM25Retriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retrieval_engine import RetrievalEngine
from src.query.query_transformer import QueryTransformer
from src.query.enhanced_retrieval import EnhancedRetriever
from src.generation.generator import RAGGenerator
from src.generation.hallucination_check import HallucinationDetector
from src.pipeline import RAGPipeline
from src.evaluation.metrics import RAGMetrics
from src.evaluation.eval_dataset import build_sample_dataset
from src.evaluation.ragas_eval import EvaluationRunner
from src.evaluation.experiment_tracker import ExperimentTracker


def build_pipeline(chunk_strategy: str, chunk_size: int):
    """Build a complete RAG pipeline with given config."""
    db_url = os.getenv("DATABASE_URL")

    # Ingest with this strategy
    ingestion = IngestionPipeline(
        chunk_strategy = chunk_strategy,
        chunk_size = chunk_size,
        db_connection = db_url,
    )

    ingestion.ingest_directory("./documents")

    # Build retrieval 
    embedder = ingestion.embedder
    store = ingestion.store

    bm25 = BM25Retriever()
    bm25.load_from_vector_store(store)

    reranker = CrossEncoderReranker()

    retrieval_engine = RetrievalEngine(
        vector_store = store,
        bm25_retriever = bm25,
        embedding_service = embedder,
        reranker = reranker,
    )

    transformer = QueryTransformer()
    enhanced = EnhancedRetriever(
        retrieval_engine = retrieval_engine,
        query_transformer = transformer,
        embedding_service = embedder,
    )

    generator = RAGGenerator()
    detector = HallucinationDetector()

    return RAGPipeline(
        enhanced_retriever = enhanced,
        generator = generator,
        hallucination_detector = detector,
    )

if __name__ == "__main__":
    dataset = build_sample_dataset()
    metrics = RAGMetrics()
    tracker = ExperimentTracker()

    # Define experiments
    configs = [
        {"strategy": "fixed",     "chunk_size": 256,  "label": "fixed_256"},
        {"strategy": "fixed",     "chunk_size": 512,  "label": "fixed_512"},
        {"strategy": "recursive", "chunk_size": 512,  "label": "recursive_512"},
    ]

    reports = []
    for config in configs:
        print(f"\n\n{'#'*60}")
        print(f"EXPERIMENT: {config['label']}")
        print(f"{'#'*60}")

        pipeline = build_pipeline(config["strategy"], config["chunk_size"])

        runner = EvaluationRunner(pipeline, metrics, dataset)
        report = runner.run(config_label = config["label"])
        runner.save_report(report)
        reports.append(config["label"])

    # Compare all experiments
    print("\n\n")
    tracker.print_comparison(reports)
