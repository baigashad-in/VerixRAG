"""
Generate evaluation Q&A pairs from documents using an LLM.

Reads each document, asks the LLM to generate diverse questions
at varying difficulty levels, then saves them as a JSON dataset
for the test runner.
"""

import json
import os
import re
import time
import argparse
from pathlib import Path
from litellm import completion
from dotenv import load_dotenv

load_dotenv()

CATEGORIES = {
    "easy": "Simple factual questions with answers clearly stated in one sentence.",
    "medium": "Questions requiring synthesis of 2-3 facts from the same section.",
    "hard": "Questions requiring information from multiple sections or inference.",
    "edge": "Tricky questions that are partially answerable, ambiguous, or test boundaries.",
    "out_of_scope": "Questions completely unrelated to the document content.",
}

RATE_LIMITS = {
    "groq": {"min_delay": 15, "max_retries": 5},
    "gemini": {"min_delay": 5, "max_retries": 5},
    "default": {"min_delay": 3, "max_retries": 3},
}

def get_provider(model: str) -> str:
    if "/" in model:
        return model.split("/")[0]
    return "default"

def parse_retry_delay(error_msg: str) -> float | None:
    match = re.search(r'retry in (\d+\.?\d*)s', str(error_msg))
    if match:
        return float(match.group(1))
    match = re.search(r'try again in (\d+)m(\d+\.?\d*)s', str(error_msg))
    if match:
        return int(match.group(1)) * 60 + float(match.group(2))
    match = re.search(r'try again in (\d+)ms', str(error_msg))
    if match:
        return int(match.group(1)) / 1000.0
    match = re.search(r'try again in (\d+)h(\d+)m', str(error_msg))
    if match:
        return int(match.group(1)) * 3600 + int(match.group(2)) * 60
    return None

def call_with_retry(model, messages, max_retries = 5, min_delay = 5.0):
    for attempt in range(max_retries + 1):
        try:
            response = completion(model = model, messages = messages, temperature = 0.7)
            return response.choices[0].message.content.strip()
        except Exception as e:
            error_str = str(e)
            is_rate_limit = "rate" in error_str.lower() or "429" in error_str or "quota" in error_str.lower()
            if not is_rate_limit:
                print(f"\n Non-rate-limit error: {error_str[:200]}")
                return None
            
            retry_delay = parse_retry_delay(error_str)
            is_daily = "per day" in error_str.lower() or "PerDay" in error_str

            if is_daily and retry_delay and retry_delay > 600:
                wait = min(retry_delay + 5, 300)
                print(f"\n DAILY LIMIT - waiting {wait:.0f}s...")
                time.sleep(wait)
                continue
            elif retry_delay:
                print(f" (wait {retry_delay:.0f}s)", end="", flush = True)
                time.sleep(retry_delay + 2)
                continue
            else:
                wait_time = min_delay * (2 ** attempt)
                print(f" (backoff {wait_time:.0f}s)", end="", flush = True)
                time.sleep(wait_time)
    print(f"\n FAILED after {max_retries} retries")
    return None


def generate_questions(document_text: str, source_file: str,
                       category: str, description: str,
                       count: int = 10, model: str = "gemini/gemini-2.0-flash",
                       min_delay: float = 5.0) -> list[dict]:
    """Generate Q&A pairs for a single category."""

    prompt = f"""You are generating evaluation questions for a RAG (Retrieval Augmented Generation) system.

DOCUMENT:
{document_text[:8000]}

SOURCE FILE: {source_file}

CATEGORY: {category}
DESCRIPTION: {description}

Generate exactly {count} questions for this category.

For each question, provide:
- The question
- The expected answer (based ONLY on the document)
- Which section(s) of the document contain the answer
- Difficulty: easy, medium, hard, or edge

For "out_of_scope" questions, the expected answer should be "OUT_OF_SCOPE" and sections should be empty.

Return ONLY a JSON array, no other text:
[
  {{
    "question": "...",
    "expected_answer": "...",
    "sections": ["..."],
    "difficulty": "...",
    "source_file": "{source_file}"
  }}
]"""

    provider = get_provider(model)
    limits = RATE_LIMITS.get(provider, RATE_LIMITS["default"])

    raw = call_with_retry(
        model = model,
        messages = [{"role": "user", "content": prompt}],
        max_retries = limits["max_retries"],
        min_delay = min_delay,
    )

    if not raw:
        return []
    try:
        # Strip markdown fences
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]

        questions = json.loads(raw)

        if not isinstance(questions, list):
            print(f"  WARNING: Expected list, got {type(questions)}")
            return []

        # Validate and clean
        valid = []
        for q in questions:
            if isinstance(q, dict) and "question" in q and "expected_answer" in q:
                q["source_file"] = source_file
                q["category"] = category
                valid.append(q)

        return valid

    except Exception as e:
        print(f"  ERROR generating {category} questions: {e}")
        return []


def generate_dataset(documents_dir: str, output_file: str,
                     questions_per_category: int = 10,
                     model: str = "gemini/gemini-2.0-flash"):
    """Generate a full evaluation dataset from all documents."""

    provider = get_provider(model)
    limits = RATE_LIMITS.get(provider, RATE_LIMITS["default"])
    min_delay = limits["min_delay"]

    doc_dir = Path(documents_dir)
    all_questions = []
    
    # Load all documents
    doc_files = list(doc_dir.glob("*.md")) + list(doc_dir.glob("*.txt"))
    print(f"Found {len(doc_files)} documents in {documents_dir}")

    progress_file = Path(output_file).with_suffix(".progress.json")
    completed_keys = set()

    if progress_file.exists():
        with open(progress_file, "r", encoding="utf-8") as f:
            progress = json.load(f)
            all_questions = progress.get("questions", [])
            completed_keys = set(progress.get("completed", []))
        print(f"Resuming: {len(completed_keys)} batches done, {len(all_questions)} questions so far")

    for doc_path in doc_files:
        print(f"\n{'='*60}")
        print(f"Processing: {doc_path.name}")
        print(f"{'='*60}")

        content = doc_path.read_text(encoding="utf-8")
        print(f"  Content length: {len(content)} chars")

        for category, description in CATEGORIES.items():
            # Skip check
            key = f"{doc_path.name}:{category}"
            if key in completed_keys:
                print(f"  SKIP {category} (already done)")
                continue

            count = questions_per_category
            # Fewer out-of-scope questions per doc
            if category == "out_of_scope":
                count = max(3, questions_per_category // 3)

            print(f"  Generating {count} {category} questions...")

            questions = generate_questions(
                document_text=content,
                source_file=doc_path.name,
                category=category,
                description=description,
                count=count,
                model=model,
                min_delay=min_delay,
            )

            all_questions.extend(questions)
            print(f"    Got {len(questions)} questions")

            # 
            if len(questions) > 0:
                completed_keys.add(key)

            progress_data = {"completed": list(completed_keys), "questions": all_questions}
            with open(progress_file, "w", encoding = "utf-8") as f:
                json.dump(progress_data, f, indent = 2, ensure_ascii = False)

            # Rate limiting
            time.sleep(min_delay)

    # Add cross-document questions
    if len(doc_files) > 1:
        print(f"\n{'='*60}")
        print("Generating cross-document questions...")
        print(f"{'='*60}")
        
        cross_doc_questions = generate_cross_document_questions(
            doc_dir, doc_files, model, questions_per_category
        )
        all_questions.extend(cross_doc_questions)

    # Save dataset
    dataset = {
        "metadata": {
            "total_questions": len(all_questions),
            "documents": [f.name for f in doc_files],
            "generated_with": model,
            "categories": {
                cat: len([q for q in all_questions if q.get("category") == cat])
                for cat in CATEGORIES
            },
        },
        "questions": all_questions,
    }

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    if progress_file.exists():
        progress_file.unlink()

    print(f"\n{'='*60}")
    print(f"Dataset saved to {output_file}")
    print(f"Total questions: {len(all_questions)}")
    for cat, desc in CATEGORIES.items():
        count = len([q for q in all_questions if q.get("category") == cat])
        print(f"  {cat}: {count}")
    print(f"{'='*60}")

    return dataset


def generate_cross_document_questions(doc_dir, doc_files, model, count=5):
    """Generate questions that require info from multiple documents."""

    # Read snippets from each doc
    snippets = {}
    for f in doc_files:
        content = f.read_text(encoding="utf-8")
        snippets[f.name] = content[:3000]

    combined = "\n\n---\n\n".join(
        f"FILE: {name}\n{text}" for name, text in snippets.items()
    )

    prompt = f"""You have access to multiple documents. Generate {count} questions
that would require information from at least 2 different documents to answer fully.

DOCUMENTS:
{combined}

Return ONLY a JSON array:
[
  {{
    "question": "...",
    "expected_answer": "...",
    "sections": ["doc1.md: section", "doc2.md: section"],
    "difficulty": "hard",
    "source_file": "cross_document",
    "category": "hard"
  }}
]"""
    
    raw = call_with_retry(
        model = model,
        messages = [{"role": "user", "content": prompt}],
        max_retries = 3,
        min_delay = 5.0,
    )

    if not raw:
        return []

    try:
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]

        questions = json.loads(raw)
        print(f"  Got {len(questions)} cross-document questions")
        return questions if isinstance(questions, list) else []

    except Exception as e:
        print(f"  ERROR generating cross-document questions: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Generate eval Q&A dataset")
    parser.add_argument("--dir", default="./documents",
                        help="Documents directory")
    parser.add_argument("--output", default="./experiments/eval_dataset.json",
                        help="Output JSON file")
    parser.add_argument("--per-category", type=int, default=10,
                        help="Questions per category per document")
    parser.add_argument("--model", default="gemini/gemini-2.0-flash",
                        help="LLM model for generation")

    args = parser.parse_args()
    generate_dataset(args.dir, args.output, args.per_category, args.model)


if __name__ == "__main__":
    main()