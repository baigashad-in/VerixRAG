"""
CLI script to run RAGAS evaluation.

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py --dataset ./eval_data.json --label recursive_512
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.evaluation.eval_dataset import EvalDataset, build_sample_dataset
from src.evaluation.metrics import RAGMetrics
from src.evaluation.ragas_eval import EvaluationRunner

def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on VerixRAG")
    parser.add_argument("--dataset", default=None, help="Path to eval dataset JSON")
    parser.add_argument("--label", default="default", help="Label for this experiment run")
    args = parser.parse_args()

    # Load or build dataset
    if args.dataset:
        dataset = EvalDataset().load(args.dataset)
    else:
        print("No dataset provided - using sample dataset")
        dataset = build_sample_dataset()

    # Build pipeline (reuses the same setup as ingest)
    from src.ingestion.embedding_service import EmbeddingService
    from src.storage.vector_store import VectorStore
    from src.retrieval.sparse_retrieval import BM25Retriever
    from src.retrieval.reranker import CrossEncoderReranker
    from src.retrieval.retrieval_engine import RetrievalEngine
    from src.query.query_transformer import QueryTransformer
    from src.query.enhanced_retrieval import EnhancedRetriever
    from src.generation.generator import RAGGenerator
    from src.generation.hallucination_check import HallucinationDetector
    from src.pipeline import RAGPipeline

    db_url = os.getenv("DATABASE_URL")

    embedder = EmbeddingService()
    store = VectorStore(connection_string=db_url, embedding_dimension=embedder.dimension)

    bm25 = BM25Retriever()
    bm25.load_from_vector_store(store)

    reranker = CrossEncoderReranker()
    engine = RetrievalEngine(store, bm25, embedder, reranker)

    transformer = QueryTransformer()
    enhanced = EnhancedRetriever(engine, transformer, embedder)

    generator = RAGGenerator()
    detector = HallucinationDetector()

    pipeline = RAGPipeline(enhanced, generator, detector)

    # Run evaluation
    metrics = RAGMetrics()
    runner = EvaluationRunner(pipeline, metrics, dataset)
    report = runner.run(config_label=args.label)
    runner.save_report(report)

if __name__ == "__main__":
    main()