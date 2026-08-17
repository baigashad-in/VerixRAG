"""Tests for domain-based source scoping."""

import json
import os
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from src.retrieval.scoping import DomainConfig, DomainScorer, load_domain_config
from src.retrieval.confidence import ConfidenceGate
from src.tests import fakes
from src.tests.conftest import FIXTURE_KB

ALIASES = {"store": ["verixstore"], "clouddeck": ["clouddeck", "cloud deck"],
           "hr": ["nexora"]}


def r(score, domain, cid=0):
    return SimpleNamespace(chunk_id=cid, rerank_score=score, metadata={"domain": domain},
                           final_score=None, domain_penalty=0.0)


# ── config loading ──

def test_load_config_formats(tmp_path):
    full = tmp_path / "a.json"
    full.write_text(json.dumps({"files": {"x.md": "Store"},
                                "domains": {"store": {"aliases": ["VerixStore"]}}}))
    cfg = load_domain_config(str(full))
    assert cfg.domain_for("x.md") == "store" and cfg.aliases["store"] == ["VerixStore"]

    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"sources": [{"filename": "y.md", "domain": "ops"}]}))
    assert load_domain_config(str(manifest)).domain_for("y.md") == "ops"

    flat = tmp_path / "f.json"
    flat.write_text(json.dumps({"z.md": "hr"}))
    assert load_domain_config(str(flat)).domain_for("z.md") == "hr"

    assert load_domain_config(str(tmp_path / "missing.json")).files == {}


def test_front_matter_overrides_mapping():
    cfg = DomainConfig(files={"x.md": "store"})
    assert cfg.domain_for("x.md", {"domain": "HR"}) == "hr"
    assert cfg.domain_for("unknown.md") is None


def test_repo_config_maps_all_project_documents():
    cfg = load_domain_config(os.path.join(os.path.dirname(FIXTURE_KB), "..", "..", "..",
                                          "config", "domains.json"))
    for name in os.listdir(FIXTURE_KB):
        if name.endswith(".md"):
            assert cfg.domain_for(name), f"{name} has no domain in config/domains.json"


# ── detection ──

def test_alias_detection():
    s = DomainScorer(ALIASES)
    assert s.detect("What is the CloudDeck offline storage limit?", None) == \
        {"domain": "clouddeck", "method": "alias"}
    assert s.detect("Nexora PTO for new hires", None)["domain"] == "hr"


def test_alias_must_be_whole_word():
    s = DomainScorer({"hr": ["hr"]})
    assert s.detect("What are the three plans?", None)["method"] != "alias"


def test_two_aliases_means_cross_domain_no_scoping():
    d = DomainScorer(ALIASES).detect("Does CloudDeck use the same rules as Nexora?", None)
    assert d["domain"] is None and d["method"] == "alias_multiple"


def test_centroid_detection_needs_margin():
    s = DomainScorer({}, min_margin=0.05)
    s.refresh({"store": [1, 0, 0], "hr": [0, 1, 0]})
    assert s.detect("q", [0.9, 0.1, 0])["domain"] == "store"
    uncertain = s.detect("q", [0.5, 0.49, 0])
    assert uncertain["domain"] is None and uncertain["method"] == "centroid_uncertain"


def test_disabled_scorer():
    assert DomainScorer(ALIASES, enabled=False).detect("CloudDeck", None)["domain"] is None


# ── applying the penalty ──

def test_penalty_reorders_marginal_off_domain_chunks():
    results = [r(3.0, "clouddeck", 1), r(2.5, "store", 2), r(-1.0, "store", 3)]
    out = DomainScorer(penalty=2.0).apply(results, {"domain": "store"})
    assert [x.chunk_id for x in out] == [2, 1, 3]
    assert out[1].final_score == 1.0 and out[1].domain_penalty == 2.0
    assert out[0].final_score == 2.5                       # in-domain untouched


def test_strong_off_domain_chunk_survives_penalty():
    results = [r(8.0, "verixapi", 1), r(3.0, "clouddeck", 2)]
    out = DomainScorer(penalty=2.0).apply(results, {"domain": "clouddeck"})
    assert out[0].chunk_id == 1                            # 6.0 still beats 3.0


def test_no_domain_detected_changes_nothing():
    results = [r(1.0, "hr", 1), r(2.0, "store", 2)]
    out = DomainScorer().apply(results, {"domain": None})
    assert [x.chunk_id for x in out] == [2, 1] and all(x.domain_penalty == 0 for x in out)


def test_untagged_chunks_never_penalised():
    out = DomainScorer().apply([r(1.0, None, 1)], {"domain": "store"})
    assert out[0].final_score == 1.0


def test_gate_ignores_penalty():
    results = DomainScorer(penalty=5.0).apply([r(0.5, "hr")], {"domain": "store"})
    assert results[0].final_score == -4.5
    assert ConfidenceGate(0.0).check(results).passed


# ── engine integration (fakes) ──

def test_engine_promotes_in_domain_phone_number(fake_models):
    docs = [
        ("Support phone number: 1-888-555-DECK (CloudDeck Enterprise only)",
         {"source": "product_manual.md", "domain": "clouddeck"}),
        ("Contact VerixStore support at 1-800-555-0199",
         {"source": "refund_policy.md", "domain": "store"}),
    ]
    q = "What is the VerixStore support phone number?"

    plain_engine = fakes.build_engine(docs)[0]
    without = plain_engine.retrieve(q, top_k=2)
    assert without[0].metadata["domain"] == "clouddeck"        # the original bug

    engine = fakes.build_engine(docs)[0]
    engine.domain_scorer = DomainScorer({"store": ["verixstore"]}, penalty=3.0)
    engine.refresh_domains()
    scoped = engine.retrieve(q, top_k=2)
    assert scoped[0].metadata["domain"] == "store"
    assert scoped[1].domain_penalty == 3.0
    assert scoped[1].rerank_score > scoped[0].rerank_score    # raw scores unchanged


# ── ingestion tags domains (pgvector) ──

@pytest.mark.db
def test_ingest_tags_domains_and_centroids(clean_db, fake_models, tmp_path, monkeypatch):
    from src.config import reset_settings
    from src.ingestion.ingest import IngestionPipeline

    kb = tmp_path / "kb"
    shutil.copytree(FIXTURE_KB, kb)
    monkeypatch.setenv("DOMAINS_CONFIG", str(kb / "_domains.json"))
    reset_settings()

    pipe = IngestionPipeline(db_connection=clean_db, documents_root=str(kb))
    results = {x["file"]: x for x in pipe.ingest_directory(str(kb))}
    assert results["product_manual.md"]["domain"] == "clouddeck"
    centroids = pipe.store.domain_centroids()
    assert set(centroids) == {"store", "clouddeck", "verixapi", "hr"}
    assert len(centroids["store"]) == fakes.DIM

    # Remapping a file re-ingests only that file
    cfg = json.loads((kb / "_domains.json").read_text())
    cfg["files"]["hr_handbook.md"] = "people"
    (kb / "_domains.json").write_text(json.dumps(cfg))
    reset_settings()
    pipe2 = IngestionPipeline(db_connection=clean_db, documents_root=str(kb))
    statuses = {x["file"]: x["status"] for x in pipe2.ingest_directory(str(kb))}
    assert statuses["hr_handbook.md"] == "ingested"
    assert sum(s == "unchanged" for s in statuses.values()) == 6
    pipe.store.close(); pipe2.store.close()


def test_calibration_domain_table():
    from scripts.calibrate import domain_table
    rows = [
        {"expected": "store", "method": "alias", "domain": "store", "centroid_top": None, "margin": None},
        {"expected": "store", "method": "centroid", "domain": "store", "centroid_top": "store", "margin": 0.10},
        {"expected": "hr", "method": "centroid_uncertain", "domain": None, "centroid_top": "store", "margin": 0.02},
    ]
    table = {row["min_margin"]: row for row in domain_table(rows, margins=(0.0, 0.05))}
    assert table[0.0]["coverage"] == 1.0 and table[0.0]["wrong"] == 1
    assert table[0.05]["coverage"] == round(2 / 3, 3) and table[0.05]["accuracy"] == 1.0
