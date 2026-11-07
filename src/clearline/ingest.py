"""Turning files into access-labelled, embedded chunks.

Markdown is split on headings first so each chunk knows its section, then long sections are
packed into windows of roughly `max_chars` with a little overlap, breaking on paragraph and
sentence boundaries. Every chunk inherits its document's classification and groups, so access
is enforced per chunk at query time.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from pypdf import PdfReader

from .models import Chunk, Classification, Document

HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.MULTILINE)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def document_id(title: str, source: str) -> str:
    return hashlib.sha256(f"{source}\x00{title}".encode()).hexdigest()[:16]


def split_sections(text: str) -> list[tuple[str, str]]:
    """(section heading, body) pairs; text before the first heading is the 'Overview'."""
    sections: list[tuple[str, str]] = []
    matches = list(HEADING.finditer(text))
    if not matches or matches[0].start() > 0:
        lead = text[: matches[0].start()] if matches else text
        if lead.strip():
            sections.append(("Overview", lead.strip()))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        if body:
            sections.append((match.group(2).strip(), body))
    return sections


def pack(body: str, max_chars: int, overlap: int) -> list[str]:
    """Pack paragraphs (then sentences, for long ones) into windows of at most `max_chars`."""
    pieces: list[str] = []
    for paragraph in re.split(r"\n\s*\n", body):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            pieces.append(paragraph)
        else:
            pieces.extend(s for s in SENTENCE_END.split(paragraph) if s)
    windows: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) + 2 > max_chars:
            windows.append(current)
            tail = current[-overlap:] if overlap else ""
            # Start the overlap on a word boundary.
            current = (tail[tail.find(" ") + 1 :] + "\n\n" if " " in tail else "") + piece
        else:
            current = f"{current}\n\n{piece}" if current else piece
        while len(current) > max_chars:  # a single sentence longer than the window
            windows.append(current[:max_chars])
            current = current[max_chars - overlap :]
    if current.strip():
        windows.append(current)
    return windows


def chunk_document(doc: Document, max_chars: int = 900, overlap: int = 120) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section, body in split_sections(doc.text):
        for text in pack(body, max_chars, overlap):
            chunks.append(Chunk(
                id=f"{doc.id}:{len(chunks)}", document_id=doc.id, title=doc.title, section=section, text=text,
                classification=doc.classification, groups=doc.groups, position=len(chunks),
            ))
    return chunks


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Minimal `key: value` front matter (title, classification, groups) at the top of a file."""
    match = FRONT_MATTER.match(text)
    if not match:
        return {}, text
    meta = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip().lower()] = value.strip()
    return meta, text[match.end() :]


def load_file(path: Path) -> Document:
    """A Markdown, text or PDF file as a Document; access labels come from front matter."""
    if path.suffix.lower() == ".pdf":
        raw = "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
        meta: dict[str, str] = {}
    else:
        meta, raw = parse_front_matter(path.read_text(encoding="utf-8"))
    title = meta.get("title") or path.stem.replace("-", " ").replace("_", " ").title()
    groups = frozenset(g.strip().lower() for g in meta.get("groups", "").split(",") if g.strip())
    return Document(
        id=document_id(title, path.name), title=title, text=raw,
        classification=Classification.parse(meta.get("classification", "unclassified")),
        groups=groups, source=path.name,
    )


def load_directory(root: Path) -> list[Document]:
    return [load_file(p) for p in sorted(root.rglob("*")) if p.suffix.lower() in {".md", ".txt", ".pdf"}]
