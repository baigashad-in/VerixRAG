"""
Evaluation dataset: questions with ground truth answers.

THIS IS THE FOUNDATION OF EVAL PIPELINE.
Without ground truth, we're guessing. With it, you can
measure exactly where your system fails.

HOW TO BUILD THIS IN PRACTICE: 
1. Write 20-50 questions the knowledge base should answer
2. For each question, find the actual chunks that contain the answer
3. Write the ideal answer based on those chunks
4. Include edge cases: multi-hop questions, negations,
out-of-scope questions

It's the most valuable thing for RAG system. Every production RAG team
maintains a dataset like this.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path

@dataclass
class EvalExample:
    """A single evaluation example with ground truth."""
    question: str
    ground_truth_answer: str
    relevant_chunk_ids: list[int] # which chunks should be retrieved
    metadata: dict = field(default_factory = dict) # difficulty, category, etc.

class EvalDataset:
    """Manages evaluation examples.
    
    Store as JSON so non-engineers can contribute questions.
    In production, domain experts write these - they know
    what users actually ask.
    """

    def __init__(self):
        self.examples: list[EvalExample] = []

    def add(self, question: str, ground_truth: str,
            relevant_chunk_ids: list[int], **metadata):
        self.examples.append(EvalExample(
            question = question,
            ground_truth_answer = ground_truth,
            relevant_chunk_ids = relevant_chunk_ids,
            metadata = metadata
        ))
    
    def save(self, path: str):
        data = [
            {
                "question": ex.question,
                "ground_truth_answer": ex.ground_truth_answer,
                "relevant_chunk_ids": ex.relevant_chunk_ids,
                "metadata": ex.metadata
            }
            for ex in self.examples
        ]
        Path(path).write_text(json.dumps(data, indent = 2))
        print(f"Saved{len(data)} eval examples to {path}")

    def load(self, path: str):
        data = json.loads(Path(path).read_text())
        self.examples = [
            EvalExample(
                question = item["question"],
                ground_truth_answer = item["ground_truth_answer"],
                relevant_chunk_ids = item["relevant_chunk_ids"],
                metadata = item.get("metadata", {})
            )
            for item in data
        ]
        print(f"Loaded {len(self.examples)} eval examples")
        return self
    
# Example dataset for the policies document:
def build_sample_dataset() -> EvalDataset:
    dataset = EvalDataset()

    dataset.add(
        question = "What is the refund policy for purchases over 30 days old?",
        ground_truth = "After 30 days, store credit equal to the purchase price is offered. Store credit never expires.",
        relevant_chunk_ids = [2, 3],
        difficulty = "easy",
        category = "refund",
    )

    dataset.add(
        question = "How much does express shipping cost and how fast is it?",
        ground_truth = "Express shipping costs $15 and delivers within 2 business days.",
        relevant_chunk_ids = [5],
        difficulty = "easy",
        category = "shipping",
    )

    dataset.add(
        question = "If I delete my account, what happens to my store credit?",
        ground_truth = "Store credit is forfeited when you delete your account.",
        relevant_chunk_ids = [3, 7],
        difficulty = "hard",
        category = "account",
        reasoning = "Requires connecting info from two different sections",
    )

    dataset.add(
        question = "What is the weather in Tokyo?",
        ground_truth = "NOT_ANSWERABLE",
        relevant_chunk_ids = [],
        difficulty = "easy",
        category = "out_of_scope",
    )

    return dataset