"""
CLI script to ingest documents into the knowledge base.

Usage:
    python scripts/ingest_docs.py
    python scripts/ingest_docs.py --directory ./my_docs
    python scripts/ingest_docs.py --strategy semantics --chunk-size 256
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.ingestion.ingest import IngestionPipeline

def main():
    parser = argparse.ArgumentParser(description="Ingest documents into VerixRAG")
    parser.add_argument("--directory", default="./docs", help="Path to document folder")
    parser.add_argument("--strategy", default="recursive", choices=["fixed", "recursive", "semantic"])
    parser.add_argument("--chunk-size", type=int, default=512)
    args = parser.parse_args()

    pipeline = IngestionPipeline(
        chunk_strategy = args.strategy,
        chunk_size = args.chunk_size,
        embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        db_connection = os.getenv("DATABASE_URL"),
    )

    results = pipeline.ingest_directory(args.directory)

    print(f"\nDone. {len(results)} files ingested.")
    for r in results:
        print(f" {r['file']}: {r['chunks_created']} chunks")

if __name__ == "__main__":
    main()
