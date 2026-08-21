"""Tests for generator output post-processing."""

from types import SimpleNamespace

import pytest

from src.generation.generator import RAGGenerator
from src.tests import fakes


def _chunk(i: int, source: str) -> SimpleNamespace:
    return SimpleNamespace(content=f"Chunk {i} text about {source}.", chunk_id=i,
                           metadata={"filename": source, "source": source, "section": ""})


@pytest.fixture
def generator(monkeypatch):
    gen = RAGGenerator()
    gen.reply = ""
    monkeypatch.setattr(gen, "_call_with_swap", lambda messages: fakes.fake_completion(gen.reply))
    return gen


CHUNKS = [_chunk(1, "shipping_policy.md"), _chunk(2, "hr_handbook.md"),
          _chunk(3, "policies.md"), _chunk(4, "hr_handbook.md")]


def test_only_cited_sources_are_returned(generator):
    generator.reply = "Express shipping costs $15 [Source 1]. It ships in 2 days [Source 3]."
    response = generator.generate("How much is express shipping?", CHUNKS)
    assert [c.citation_id for c in response.citations] == [1, 3]
    assert {c.source_file for c in response.citations} == {"shipping_policy.md", "policies.md"}


def test_grouped_and_ranged_citations(generator):
    generator.reply = "Refunds take 30 days [Sources 1 and 3]. Details vary [Source 2-4]."
    ids = [c.citation_id for c in generator.generate("q", CHUNKS).citations]
    assert ids == [1, 2, 3, 4]


def test_refusal_has_no_citations(generator):
    generator.reply = "I don't have enough information in the available documents to answer this question."
    assert generator.generate("q", CHUNKS).citations == []


def test_typographic_hyphens_and_spaces_are_normalised(generator):
    generator.reply = "Error code E\u20114012 means a carrier exception\u202fcall support [Source 1]."
    answer = generator.generate("error code E-4012", CHUNKS).answer
    assert "E-4012" in answer and "\u202f" not in answer
    generator.reply = "Employees with 0\u20132 years get 15 days [Source 1]."     # en dash kept
    assert "0\u20132" in generator.generate("q", CHUNKS).answer