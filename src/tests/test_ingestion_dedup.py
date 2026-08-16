"""Tests for chunk deduplication, document metadata and idempotent ingestion."""

import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from src.ingestion.dedup import Deduplicator, fingerprint, jaccard, normalize, word_set
from src.ingestion.metadata import (
    ChunkLocator, document_title, ingest_signature, parse_front_matter, section_path,
    heading_index,
)
from src.tests.conftest import FIXTURE_KB


def chunk(content, source="a.md", idx=0, vec=None):
    if vec is None:
        rng = np.random.default_rng(abs(hash(content)) % 2**32)
        vec = rng.normal(size=8)
    return SimpleNamespace(content=content, embedding=list(vec),
                           metadata={"source": source, "chunk_index": idx})


# ── normalisation & similarity ──

def test_normalize_ignores_case_whitespace_markdown():
    assert normalize("## Refund  Policy\n\n**All** purchases") == normalize("refund policy all purchases")
    assert fingerprint("# Title\nText") == fingerprint("title   text")


def test_jaccard_keeps_different_numbers_apart():
    a = word_set("Business plan: 5000 requests per minute")
    b = word_set("Standard plan: 1000 requests per minute")
    assert jaccard(a, b) < 0.85


# ── deduplicator ──

def test_exact_duplicate_dropped_first_copy_kept():
    d = Deduplicator()
    report = d.filter([chunk("Store credit never expires.", "policies.md", 0),
                       chunk("STORE credit   never expires.", "refund_policy.md", 3)])
    assert [c.metadata["source"] for c in report.kept] == ["policies.md"]
    assert report.dropped[0]["reason"] == "exact"
    assert report.dropped[0]["duplicate_of"] == "policies.md#chunk0"
    assert report.kept[0].metadata["chunk_fp"]


def test_near_duplicate_needs_both_cosine_and_word_overlap():
    v = np.ones(8)
    near_text_a = "All purchases are eligible for a full refund within 30 days of purchase receipt required"
    near_text_b = "All purchases are eligible for a full refund within 30 days of the purchase, receipt required."
    different = "Express shipping costs $15 and arrives within 2 business days"
    report = Deduplicator(0.95, 0.85).filter([
        chunk(near_text_a, "a.md", 0, v),
        chunk(near_text_b, "b.md", 0, v * 1.01),     # same direction -> cosine 1.0
        chunk(different, "c.md", 0, v),              # cosine 1.0 but different words
    ])
    # b is the more complete copy (one extra word), so it wins over a
    assert [c.metadata["source"] for c in report.kept] == ["b.md", "c.md"]
    assert report.dropped[0]["reason"] == "near"
    assert report.dropped[0]["duplicate_of"] == "b.md#chunk0"


def test_most_complete_copy_kept_regardless_of_file_order():
    v = np.ones(8)
    truncated = "Express shipping is available for an additional and delivers within 2 business days"
    full = "Express shipping is available for an additional $15 and delivers within 2 business days"
    report = Deduplicator(0.95, 0.85).filter([
        chunk(truncated, "policies.md", 0, v),            # sorts first by name
        chunk(full, "shipping_policy.md", 0, v),
    ])
    assert [c.metadata["source"] for c in report.kept] == ["shipping_policy.md"]
    assert "$15" in report.kept[0].content


def test_external_lookup_checks_stored_chunks():
    stored = [{"id": 7, "content": "Store credit never expires.", "source": "policies.md",
               "similarity": 0.99}]
    report = Deduplicator().filter([chunk("store credit never expires", "refund_policy.md")],
                                   external_lookup=lambda emb: stored)
    assert not report.kept
    assert report.dropped[0]["reason"] == "exact_stored"
    assert report.dropped[0]["duplicate_of"] == "policies.md#id7"


# ── metadata ──

DOC = """# Account Management Guide

## Creating an Account

Use a strong password.

## Password Reset

### Email link

The link expires after 24 hours.
"""


def test_title_and_section_paths():
    assert document_title(DOC, "x.md") == "Account Management Guide"
    assert document_title("no headings here", "x.md") == "x.md"
    headings = heading_index(DOC)
    pos = DOC.index("The link expires")
    assert section_path(headings, pos) == "Account Management Guide › Password Reset › Email link"
    assert section_path(headings, DOC.index("Use a strong")) == \
        "Account Management Guide › Creating an Account"


def test_chunk_locator_tracks_order():
    loc = ChunkLocator(DOC)
    p1, s1 = loc.locate("## Creating an Account\n\nUse a strong password.")
    p2, s2 = loc.locate("The link expires after 24 hours.")
    assert p1 < p2 and s1.endswith("Creating an Account") and s2.endswith("Email link")
    assert loc.locate("text that is not in the document") == (None, "")


def test_front_matter():
    meta, body = parse_front_matter("---\ndomain: store\nTitle: Refunds\n---\n# Refunds\n")
    assert meta == {"domain": "store", "title": "Refunds"}
    assert body.startswith("# Refunds")
    assert parse_front_matter("# No front matter") == ({}, "# No front matter")


def test_signature_changes_with_content_or_config():
    base = ingest_signature("text", {"chunk_size": 512})
    assert base == ingest_signature("text", {"chunk_size": 512})
    assert base != ingest_signature("text!", {"chunk_size": 512})
    assert base != ingest_signature("text", {"chunk_size": 256})


# ── end-to-end against pgvector (skipped without TEST_DATABASE_URL) ──

@pytest.fixture
def kb_copy(tmp_path):
    dst = tmp_path / "kb"
    shutil.copytree(FIXTURE_KB, dst)
    return dst


@pytest.mark.db
def test_ingestion_dedups_tags_sections_and_skips_unchanged(clean_db, fake_models, kb_copy):
    from src.ingestion.ingest import IngestionPipeline

    pipe = IngestionPipeline(db_connection=clean_db, documents_root=str(kb_copy))
    first = {r["file"]: r for r in pipe.ingest_directory(str(kb_copy))}
    assert all(r["status"] == "ingested" for r in first.values())

    # refund_policy.md has the more complete copy of the shared refund text,
    # so policies.md's copy is the one dropped
    assert first["policies.md"]["duplicates_dropped"] >= 1
    assert first["refund_policy.md"]["duplicates_dropped"] == 0
    registry = {d["source"]: d for d in pipe.store.list_documents()}
    assert registry["policies.md"]["duplicates_dropped"] >= 1
    assert registry["account_guide.md"]["domain"] == "store"
    assert pipe.last_perf["duplicates_dropped"] >= 1
    assert set(pipe.last_perf["stage_ms"]) >= {"signature", "chunk", "embed", "dedup", "store"}

    with pipe.store.conn.cursor() as cur:
        cur.execute("""SELECT metadata FROM chunks
                       WHERE metadata->>'source' = 'account_guide.md'
                       AND content LIKE '%%expires after 24 hours%%'""")
        meta = cur.fetchone()[0]
    assert meta["title"] == "Account Management Guide"
    assert meta["section"].endswith("Password Reset")
    assert meta["doc_hash"] and meta["ingest_sig"] and meta["ingested_at"]

    # Second run: nothing changed -> nothing re-embedded, including
    # policies.md whose chunks were all dropped (the registry remembers it)
    second = pipe.ingest_directory(str(kb_copy))
    assert all(r["status"] == "unchanged" for r in second)
    assert pipe.last_perf["stage_ms"]["embed"] < 50

    # Edit one file -> only that file is re-ingested
    (kb_copy / "shipping_policy.md").write_text("# Shipping Policy\n\nExpress shipping is $20.\n")
    third = {r["file"]: r["status"] for r in pipe.ingest_directory(str(kb_copy))}
    assert third["shipping_policy.md"] == "ingested"
    assert sum(s == "unchanged" for s in third.values()) == 6

    # Delete a file + prune -> its chunks disappear
    (kb_copy / "hr_handbook.md").unlink()
    fourth = {r["file"]: r["status"] for r in pipe.ingest_directory(str(kb_copy), prune=True)}
    assert fourth["hr_handbook.md"] == "pruned"
    assert "hr_handbook.md" not in pipe.store.source_signatures()
    assert "hr_handbook.md" not in {d["source"] for d in pipe.store.list_documents()}
    pipe.store.close()
