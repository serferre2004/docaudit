"""Keyword search with BM25.   >>> YOUR TURN: implement this file. <<<

Tests that define "done": tests/rag/test_keyword.py

Embeddings can tell that "settle the bill" means "pay the invoice". BM25 doesn't know any synonyms,
but an exact, rare term like ``AP-17`` carries a lot of weight, it needs no model, and you can
explain why a chunk matched. Which one retrieves better on a given corpus is something to measure.

For one query term in one chunk:

    idf   = ln(1 + (N - n + 0.5) / (n + 0.5))
    score = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / avg_length))

    N: chunks in the index        n: chunks containing the term     tf: times it occurs in the chunk
    length: tokens in the chunk   avg_length: average over the index

A chunk's score is the sum over the distinct terms of the query.
"""

import re
from collections import Counter
from collections.abc import Sequence
from math import log

from docaudit.rag.schemas import Chunk, SearchHit


def tokenize(text: str) -> list[str]:
    """Split ``text`` into lowercase terms, keeping identifiers in one piece.

    Requirements:
      * Terms are runs of letters and digits (accented letters too).
      * ``.`` and ``-`` stay only BETWEEN letters or digits, so ``"Clause 4.2, form AP-17!"`` gives
        ``["clause", "4.2", "form", "ap-17"]``. Everything else separates terms.
    """
    re_expresion = r"[^\W_]+(?:[-.][^\W_]+)*"
    return re.findall(re_expresion, text.lower())


class BM25Index:
    """An in-memory BM25 index over ``chunk.search_text``. Implements ``Retriever``.

    Requirements:
      * ``__init__(self, *, k1=1.5, b=0.75)`` starts empty.
      * ``add(chunks)`` can be called several times; statistics cover everything added so far.
      * ``search(query, k=5)``:
          - ``k < 1`` raises ``ValueError``.
          - Returns the ``k`` best ``SearchHit`` s, highest score first. Ties keep insertion order.
          - Chunks sharing no term with the query are NOT returned: unlike vector search, a score
            of zero really means "no match", so an empty result is possible.

    ``search`` does no I/O. It is ``async`` only to satisfy the ``Retriever`` protocol.
    """

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._term_counts: list[Counter[str]] = []
        self._lengths: list[int] = []
        self._doc_freq: Counter[str] = Counter()

    def add(self, chunks: Sequence[Chunk]) -> None:
        for chunk in chunks:
            self._chunks.append(chunk)
            tokens = tokenize(chunk.search_text)
            chunk_counter = Counter(tokens)
            self._term_counts.append(chunk_counter)
            self._lengths.append(len(tokens))
            self._doc_freq.update(chunk_counter.keys())

    async def search(self, query: str, k: int = 5) -> list[SearchHit]:
        if k < 1:
            raise ValueError(f"Parameter k must be greater or equal than 1, got {k}")
        query_tokens = set(tokenize(query))
        N = len(self._chunks)
        if N == 0:
            return []
        avg_length = sum(self._lengths) / N
        search_hits: list[SearchHit] = []
        iterable = zip(self._chunks, self._term_counts, self._lengths, strict=True)
        for chunk, t_count, length in iterable:
            score = 0.0
            for term in query_tokens:
                n = self._doc_freq[term]
                idf = log(1 + (N - n + 0.5) / (n + 0.5))
                tf = t_count[term]
                normalize = 1 - self.b + self.b * length / avg_length
                score += idf * tf * (self.k1 + 1) / (tf + self.k1 * normalize)
            if score > 0:
                search_hits.append(SearchHit(chunk=chunk, score=score))
        search_hits.sort(key=lambda x: -x.score)
        return search_hits[:k]
