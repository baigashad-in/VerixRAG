"""
Document metadata extraction: titles, section paths, content hashes.

A citation that says "account_guide.md" makes the reader search the
whole file. "account_guide.md › Account Management Guide › Password
Reset" points at the paragraph. Content hashes make re-ingestion
idempotent: an unchanged file is skipped instead of re-embedded,
and every chunk records which version of its document it came from.
"""

import hashlib
import json
import re
from datetime import datetime, timezone

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$", re.MULTILINE)
_FRONT_MATTER_RE = re.compile(r"\A\ufeff?---[ \t]*\n(.*?)\n---[ \t]*\n", re.DOTALL)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_hash(text: str) -> str:
    """Short, stable hash of a document's content (its version id)."""
    return sha256(text)[:16]


def ingest_signature(content: str, config: dict) -> str:
    """Changes when the content OR any setting that affects chunks changes."""
    blob = content + "\x00" + json.dumps(config, sort_keys=True, default=str)
    return sha256(blob)[:16]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_front_matter(text: str) -> tuple[dict, str]:
    """Read a simple `key: value` front-matter block, if the file has one.

    ---
    domain: store
    title: Refund Policy
    ---
    """
    match = _FRONT_MATTER_RE.match(text)
    if not match:
        return {}, text
    meta = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip().lower()
            if re.match(r"^[a-z_][a-z0-9_]*$", key):
                meta[key] = value.strip().strip("'\"")
    return meta, text[match.end():]


def document_title(text: str, fallback: str) -> str:
    for match in _HEADING_RE.finditer(text):
        if len(match.group(1)) == 1:
            return match.group(2).strip()
    return fallback


def heading_index(text: str) -> list[tuple[int, int, str]]:
    """[(char_position, level, heading_text), ...] in document order."""
    return [(m.start(), len(m.group(1)), m.group(2).strip())
            for m in _HEADING_RE.finditer(text)]


def section_path(headings: list[tuple[int, int, str]], position: int) -> str:
    """Heading path ("H1 › H2 › H3") in effect at a character position."""
    path: list[tuple[int, str]] = []
    for pos, level, title in headings:
        if pos > position:
            break
        while path and path[-1][0] >= level:
            path.pop()
        path.append((level, title))
    return " › ".join(title for _, title in path)


class ChunkLocator:
    """Finds each chunk's position in its document, in order.

    Chunkers strip and re-join text, so we search for the start of each
    chunk from where the previous one began. Falls back to a search
    from the top; returns None if the text can't be located.
    """

    def __init__(self, text: str):
        self.text = text
        self.cursor = 0
        self.headings = heading_index(text)

    def locate(self, chunk_text: str) -> tuple[int | None, str]:
        probe = chunk_text.strip()[:80]
        if not probe:
            return None, ""
        pos = self.text.find(probe, self.cursor)
        if pos < 0:
            pos = self.text.find(probe)
        if pos < 0:
            return None, ""
        self.cursor = pos
        return pos, section_path(self.headings, pos)