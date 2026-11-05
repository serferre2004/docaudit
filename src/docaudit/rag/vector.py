"""Vector search.   >>> YOUR TURN: implement this file. <<<

Tests that define "done": tests/rag/test_vector.py

No vector database: for a few hundred chunks, comparing the query against every vector in a Python
list is fast enough, and you see exactly what a vector database does before its optimizations.
"""

import math
from collections.abc import Sequence

from docaudit.rag.embeddings import Embedder
from docaudit.rag.schemas import Chunk, SearchHit


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the cosine of the angle between two vectors, from -1.0 to 1.0.

    Requirements:
      * Vectors of different lengths raise ``ValueError``: they came from different models, so
        any number returned would be meaningless.
      * If either vector is all zeros the cosine is undefined: return ``0.0``.
    """
    if len(a) != len(b):
        raise ValueError(f"vectors of different lengths: {len(a)} != {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorIndex:
    """An in-memory index that ranks chunks by cosine similarity. Implements ``Retriever``.

    Requirements:
      * ``__init__(self, embedder)`` starts empty.
      * ``add(chunks)`` embeds every ``chunk.search_text`` (NOT ``chunk.text``) with ONE call to
        ``embedder.embed`` and appends to the index. No chunks means no call: embeddings cost
        time or money.
      * ``search(query, k=5)``:
          - ``k < 1`` raises ``ValueError``.
          - Empty index returns ``[]`` without embedding the query.
          - Returns the ``k`` best ``SearchHit`` s, highest score first. Ties keep insertion order.
    """

    def __init__(self, embedder: Embedder) -> None:
        self.index: list[tuple[Chunk, list[float]]] = []
        self.embedder = embedder

    async def add(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        embeddings = await self.embedder.embed([c.search_text for c in chunks])
        self.index.extend(zip(chunks, embeddings, strict=True))

    async def search(self, query: str, k: int = 5) -> list[SearchHit]:
        if k < 1:
            raise ValueError(f"Parameter k must be greater or equal than 1, got {k}")
        if len(self.index) == 0:
            return []
        query_embedding = (await self.embedder.embed([query]))[0]
        search_hits = []
        for chunk, vector in self.index:
            score = cosine_similarity(vector, query_embedding)
            search_hits.append(SearchHit(chunk=chunk, score=score))
        search_hits.sort(key=lambda x: -x.score)
        return search_hits[:k]
