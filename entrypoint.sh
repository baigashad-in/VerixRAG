#!/bin/bash
set -e
python -m scripts.ingest_docs --prune 
exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000
