"""
CLI script to ingest documents into the knowledge base.

Usage:
    python scripts/ingest_docs.py
    python scripts/ingest_docs.py --directory ./my_docs
    python scripts/ingest_docs.py --strategy semantics --chunk-size 256
    python scripts/ingest_docs.py --force     # re-embed even unchanged files
    python scripts/ingest_docs.py --prune     # drop stored docs no longer on disk
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
    parser.add_argument("--directory", default="./documents", help="Path to documents folder")
    parser.add_argument("--strategy", default="recursive", choices=["fixed", "recursive", "semantic"])
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--force", action="store_true",
                        help="re-ingest documents even if their content is unchanged")
    parser.add_argument("--prune", action="store_true",
                        help="delete stored documents that no longer exist in --directory")
    args = parser.parse_args()

    pipeline = IngestionPipeline(
        chunk_strategy = args.strategy,
        chunk_size = args.chunk_size,
        embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        db_connection = os.getenv("DATABASE_URL"),
    )

    results = pipeline.ingest_directory(args.directory, force=args.force, prune=args.prune)

    print(f"\nDone. {len(results)} files processed.")
    for r in results:
        extra = f", {r['duplicates_dropped']} duplicates dropped" if r.get("duplicates_dropped") else ""
        print(f" {r['file']}: {r['status']}, {r['chunks_created']} chunks{extra}")

if __name__ == "__main__":
    main()
