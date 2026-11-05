"""Splitting documents into retrievable chunks.   >>> YOUR TURN: implement this file. <<<

Tests that define "done": tests/rag/test_chunking.py

Contracts already have a natural unit: the numbered clause. It is also what a citation has to
point at, so we split along clauses and only fall back to fixed-size windows when one is too long.

Expected document format:

    MASTER SERVICES AGREEMENT MSA-ALS            <- preamble: anything before the first clause
    Between Aston Logistics and DocAudit Demo.

    ## 4. Invoicing and Payment                  <- heading: applies until the next heading
    4.1 Customer shall pay within 30 days.       <- clause: "<number>.<number> <text>"
    A clause can continue on the next line.
    4.2 Overdue amounts accrue interest.
"""

import re

from docaudit.rag.schemas import Chunk, SourceDocument

CLAUSE_NUMBERING_FMT = r"(?P<section>\d+\.\d+)\s+(?P<text>.+)"
HEADING_FMT = r"##\s+(?P<text>.+)"


def chunk_document(
    document: SourceDocument, *, max_chars: int = 800, overlap: int = 100
) -> list[Chunk]:
    """Return the chunks of ``document``, in reading order.

    Requirements:
      * ``overlap`` outside ``0 <= overlap < max_chars`` raises ``ValueError``.
      * One chunk per clause, with ``section`` = its number (``"4.1"``), ``heading`` = the current
        ``## `` heading without the ``## `` (``"4. Invoicing and Payment"``; empty if none yet)
        and ``id`` = ``"<doc id>#<section>"``. Copy ``doc_id`` and ``doc_title`` from the document.
      * Lines after a clause belong to it until the next clause or heading.
      * Whitespace in ``text`` is normalized: words separated by single spaces.
      * Non-blank text before the first clause and heading becomes one chunk with
        ``section="preamble"``. Blank-only documents produce no chunks.
      * Non-blank text between a heading and its first clause becomes one chunk with
        ``section="<key>-intro"``, where ``<key>`` is the heading's first word without a trailing
        ``.`` (``"4-intro"`` for ``"4. Invoicing and Payment"``).
      * A clause longer than ``max_chars`` becomes several parts, with ids ``"<doc>#<section>-p1"``,
        ``-p2``... and the same ``section``. Each part:
          - starts with the previous part's last whole words, as many as fit in
            ``overlap`` characters, so a sentence cut at the border survives in one piece,
          - must add new words: splitting always moves forward.

    Why overlap? Without it, a fact straddling a border is in neither part in full.
    """
    if not (0 <= overlap < max_chars):
        raise ValueError(f"overlap {overlap} must be 0 <= overlap < max_chars {max_chars}")
    heading = ""
    section = "preamble"
    lines: list[str] = []
    chunks: list[Chunk] = []
    for line in document.text.splitlines():
        line = line.strip()
        if match := re.match(HEADING_FMT, line):
            chunks.extend(_build_chunks(document, section, heading, lines, max_chars, overlap))
            lines = []
            heading = match.group("text")
            section = f"{heading.split()[0].rstrip('.')}-intro"
        elif match := re.match(CLAUSE_NUMBERING_FMT, line):
            chunks.extend(_build_chunks(document, section, heading, lines, max_chars, overlap))
            lines = [match.group("text")]
            section = match.group("section")
        else:
            lines.append(line)
    chunks.extend(_build_chunks(document, section, heading, lines, max_chars, overlap))
    return chunks


def _build_chunks(
    document: SourceDocument,
    section: str,
    heading: str,
    lines: list[str],
    max_chars: int,
    overlap: int,
) -> list[Chunk]:
    words = " ".join(lines).split()
    text = " ".join(words)
    if not text:
        return []
    if len(text) <= max_chars:
        texts_and_ids = [(text, f"{document.id}#{section}")]
    else:
        parts = _split_chunk(words, max_chars, overlap)
        texts_and_ids = [
            (part, f"{document.id}#{section}-p{i}") for i, part in enumerate(parts, start=1)
        ]
    return [
        Chunk(
            id=chunk_id,
            doc_id=document.id,
            doc_title=document.title,
            section=section,
            heading=heading,
            text=part,
        )
        for part, chunk_id in texts_and_ids
    ]


def _split_chunk(words: list[str], max_chars: int, overlap: int) -> list[str]:
    parts: list[str] = []
    start = 0
    while start < len(words):
        end = start + 1
        length = len(words[start])
        while end < len(words) and length + 1 + len(words[end]) <= max_chars:
            length += 1 + len(words[end])
            end += 1
        part = words[start:end]
        parts.append(" ".join(part))
        if end == len(words):
            break
        next_start = end
        used = 0
        while next_start - 1 > start:
            cost = len(words[next_start - 1]) + (1 if used > 0 else 0)
            if used + cost > overlap or used + cost + 1 + len(words[end]) > max_chars:
                break
            used += cost
            next_start -= 1
        start = next_start
    return parts
