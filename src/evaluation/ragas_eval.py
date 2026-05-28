"""
Evaluation runner - measures RAG pipeline end-to-end.

This is the script that tells you: "your system scores 0.85
faithfulness, 0.79 answer relevancy, 0.72 context precision,
0.68 context recall." Those numbers drive every decision about
chunking strategy, embedding model, retrieval parameters. 
"""

from dataclasses import dataclass
import json
import time
from pathlib import Path
from datetime import datetime

@dataclass
class EvalResult:
    """Results for a single evaluation example."""
    question: str
    generated_answer: str
    ground_truth: str
    retrieved_contexts: list[str]
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

class EvaluationRunner:
    """Run RAGAS evaluation against pipeline.
    
    WORKFLOW:
    1. Load eval dataset (questions + ground truth)
    2. For each question, run it through pipeline
    3. Score the pipeline's output on all 4 metrics
    4. Aggregate scores and save report
    """
    def __init__(self, rag_pipeline, metrics, eval_dataset):
        """
        Args:
            rag_pipeline: your RAGPipeline instance
            metrics: RAGMetrics instance
            eval_dataset: EvalDataset with ground truth
        """
        self.pipeline = rag_pipeline
        self.metrics = metrics
        self.dataset = eval_dataset

    def run(self, config_label: str = "default") -> dict:
        """Evaluate the pipeline on all examples.
        Args:
            config_label: name for this experiment run, e.g.
                "recursive_512_bge" or "fixed_256_minilm"
                Used to compare different configurations.
            """
        results = []
        start_time = time.time()

        print(f"\nRunning evaluation: {config_label}")
        print(f"Examples: {len(self.dataset.examples)}")
        print("-" * 50)

        for i, example in enumerate(self.dataset.examples):
            print(f"\n[{i+1}/{len(self.dataset.examples)}]"
                  f"{example.question[:60]}...")
            
            # 1. Run the pipeline
            pipeline_output = self.pipeline.answer(
                query = example.question,
                check_hallucination = False, # we measure this ourselves
            )

            answer = pipeline_output["answer"]

            # 2. Get the retrieved contexts (chunk contents)
            retrieval_results = self.pipeline.retriever.retrieve(
                query = example.question, top_k = 5
            )

            contexts = [r.content for r in retrieval_results]

            # 3. Score on all 4 metrics
            faith = self.metrics.faithfulness(answer, contexts)
            print(f" Faithfulness: {faith:.3f}")

            relevancy = self.metrics.answer_relevancy(
                example.question, answer
            )
            print(f" Answer Relevancy: {relevancy:.3f}")

            precision = self.metrics.context_precision(
                example.question, contexts,
                example.ground_truth_answer
            )
            print(f" Context Precision: {precision:.3f}")

            recall = self.metrics.context_recall(
                example.ground_truth_answer, contexts
            )
            print(f" Context Recall: {recall:.3f}")

            results.append(EvalResult(
                question = example.question,
                generated_answer = answer,
                ground_truth = example.ground_truth_answer,
                retrieved_contexts = contexts,
                faithfulness = faith,
                answer_relevancy = relevancy,
                context_precision = precision,
                context_recall = recall,
            ))

        elapsed = time.time() - start_time

        # 4. Aggregate
        report = self._build_report(results, config_label, elapsed)
        self._print_report(report)

        return report
        
    def _build_report(self, results: list[EvalResult],
                      config_label: str, elapsed: float) -> dict:
        """Aggregate individual scores into a summary report."""
        n = len(results)
        if n == 0:
            return {"error": "no results"}
        
        report = {
            "config": config_label,
            "timestamp": datetime.now().isoformat(),
            "num_examples": n,
            "elapsed_seconds": round(elapsed, 1),
            "aggregate": {
                "faithfulness": round(
                    sum(r.faithfulness for r in results) / n, 4
                ),
                "answer_relevancy": round(
                    sum(r.answer_relevancy for r in results) / n, 4
                ),
                "context_precision": round(
                    sum(r.context_precision for r in results) / n, 4
                ),
                "context_recall": round(
                    sum(r.context_recall for r in results) / n, 4
                ),
            },
            "per_example": [
                {
                    "question": r.question,
                    "faithfulness": r.faithfulness,
                    "answer_relevancy": r.answer_relevancy,
                    "context_precision": r.context_precision,
                    "context_recall": r.context_recall,
                }
                for r in results
            ],
        }
        return report
    
    def _print_report(self, report: dict):
        """Print a readable summary."""
        agg = report["aggregate"]
        print(f"\n{'='*50}")
        print(f"EVALUATION REPORT:  {report['config']}")
        print(f"{'='*50}")
        print(f"Examples:           {report['num_examples']}")
        print(f"Time:               {report['elapsed_seconds']}s")
        print(f"")
        print(f"Faithfulness:       {agg['faithfulness']:.4f}")
        print(f"Answer Relevancy:   {agg['answer_relevancy']:.4f}")
        print(f"Context Precision:  {agg['context_precision']:.4f}")
        print(f"Context Recall:     {agg['context_recall']:.4f}")
        print(f"{'='*50}")

    def save_report(self, report: dict, directory: str = "./experiments"):
        """Save report to JSON for later comparison."""
        Path(directory).mkdir(parents = True, exist_ok = True)

        filename = f"{report['config']}_{report['timestamp'][:10]}.json"
        filepath = Path(directory) / filename

        filepath.write_text(json.dumps(report, indent = 2))
        print(f"Report saved : {filepath}")