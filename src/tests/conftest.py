"""
Shared test setup.

If sentence-transformers is not installed (CI, laptops without the
model stack), register a stub module backed by the deterministic
fakes so the rest of the codebase can still be imported and tested.
Tests that need deterministic behaviour patch the fakes in explicitly
via the `fake_models` fixture, so they behave the same either way.
"""

import importlib
import os
import sys
import types

import pytest
import psycopg
from psycopg import sql
from src.tests import fakes
from src.config import reset_settings

try:
    importlib.import_module("sentence_transformers")
except ImportError:
    stub = types.ModuleType("sentence_transformers")
    stub.SentenceTransformer = fakes.HashSentenceTransformer
    stub.CrossEncoder = fakes.FakeCrossEncoder
    sys.modules["sentence_transformers"] = stub

# Pin every tunable setting to its default, overriding the developer's
# shell or .env (env_file puts .env into the container's environment).
# Tests that need other values set them with monkeypatch.setenv.
TEST_DEFAULTS = {
    "RERANK_MIN_SCORE": "-6.0",
    "SCOPING_ENABLED": "true", 
    "DOMAIN_PENALTY": "2.0", 
    "DOMAIN_MIN_MARGIN": "0.05",
    "DOMAINS_CONFIG": "config/domains.json",
    "DEDUP_ENABLED": "true", 
    "DEDUP_COSINE": "0.95", 
    "DEDUP_JACCARD": "0.85",
    "CACHE_ENABLED": "true", 
    "CACHE_TTL_SECONDS": "3600", 
    "CACHE_MAX_ENTRIES": "512",
    "TRACE_BUFFER_SIZE": "500", 
    "TRACE_TRACEMALLOC": "false",
    "TRACE_LOG_PATH": "",                     # never write trace files from tests
    "CLASSIFIER_TIMEOUT": "5", 
    "CLASSIFIER_RETRIES": "1",
    "KEYWORD_SCOPE_ENABLED": "true", 
    "KEYWORD_SCOPE_BODY_TERMS": "10",
    "KEYWORD_SCOPE_MIN_MATCHES": "2",
}
os.environ.update(TEST_DEFAULTS)

FIXTURE_KB = os.path.join(os.path.dirname(__file__), "fixtures", "kb")


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: needs live LLM API keys and an ingested DB")
    config.addinivalue_line("markers", "db: needs a pgvector database at TEST_DATABASE_URL")


def pytest_collection_modifyitems(config, items):
    """Integration tests call live LLMs and need an ingested DB: opt in only.

    RUN_INTEGRATION=1 pytest -m integration
    """
    if os.getenv("RUN_INTEGRATION") == "1":
        return
    skip = pytest.mark.skip(reason="integration test: set RUN_INTEGRATION=1 to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _fresh_settings():
    reset_settings()
    yield
    reset_settings()


@pytest.fixture
def fake_models(monkeypatch):
    """Use the deterministic fakes even if the real libraries are installed."""
    import src.ingestion.embedding_service as es
    monkeypatch.setattr(es, "SentenceTransformer", fakes.HashSentenceTransformer)
    monkeypatch.setattr(sys.modules["sentence_transformers"], "CrossEncoder",
                        fakes.FakeCrossEncoder, raising=False)
    return fakes


@pytest.fixture
def db_url():
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set (needs Postgres with pgvector)")
    # These tests DROP the chunks and documents tables. Refuse to touch a
    # database that isn't clearly a test one, e.g. the real ragdb.
    dbname = psycopg.conninfo.conninfo_to_dict(url).get("dbname", "")
    if "test" not in dbname.lower():
        pytest.exit(f"TEST_DATABASE_URL points at database '{dbname}'; its name must "
                    f"contain 'test' (these tests drop tables)", returncode=2)
    _ensure_database(url, dbname)
    return url


def _ensure_database(url: str, dbname: str) -> None:
    """Create the test database if it does not exist."""
    try:
        psycopg.connect(url).close()
        return
    except psycopg.OperationalError as exc:
        if "does not exist" not in str(exc):
            raise
    admin = psycopg.conninfo.make_conninfo(url, dbname="postgres")
    with psycopg.connect(admin, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))


@pytest.fixture
def clean_db(db_url):
    """Empty chunks table before and after the test."""
    with psycopg.connect(db_url) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("DROP TABLE IF EXISTS chunks")
        conn.execute("DROP TABLE IF EXISTS documents")
        conn.commit()
    yield db_url
    with psycopg.connect(db_url) as conn:
        conn.execute("DROP TABLE IF EXISTS chunks")
        conn.execute("DROP TABLE IF EXISTS documents")
        conn.commit()
