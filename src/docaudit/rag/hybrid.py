"""Hybrid search: combining rankings.   >>> YOUR TURN: implement this file. <<<

Tests that define "done": tests/rag/test_hybrid.py

A cosine of 0.8 and a BM25 score of 7.3 are in different units: adding them means nothing.
Reciprocal Rank Fusion ignores scores and uses only positions, so any rankings can be combined.
"""

from collections.abc import Sequence

from docaudit.rag.base import Retriever
from docaudit.rag.schemas import Chunk, SearchHit


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[SearchHit]], *, constant: int = 60
) -> list[SearchHit]:
    """Merge several rankings into one.

    Requirements:
      * A chunk's score is the sum, over the rankings where it appears, of
        ``1 / (constant + rank)``, with ``rank`` starting at 1.
      * Chunks are identified by ``chunk.id``; each appears once in the result.
      * Return ``SearchHit`` s carrying the new score, highest first. Ties keep the order of first
        appearance.
    """
    chunks: dict[str, Chunk] = {}
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            chunk_id = hit.chunk.id
            chunks.setdefault(chunk_id, hit.chunk)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (constant + rank)

    search_hits = [SearchHit(chunk=chunks[i], score=score) for i, score in scores.items()]
    search_hits.sort(key=lambda x: -x.score)
    return search_hits


class HybridRetriever:
    """Fuses several retrievers. Implements ``Retriever``.

    Requirements:
      * ``__init__(self, retrievers, *, candidates=20)``: an empty ``retrievers`` raises
        ``ValueError``.
      * ``search(query, k=5)``: ``k < 1`` raises ``ValueError``. Asks each retriever for
        ``max(candidates, k)`` hits, fuses them with ``reciprocal_rank_fusion`` and returns the
        top ``k``.

    Why ask for more than ``k``? A chunk ranked 8th by both retrievers can beat one ranked 1st by
    only one of them, but only if fusion gets to see it. More candidates also let in more noise
    from a weak retriever, so ``candidates`` is a parameter to measure, not a "bigger is better".
    """

    def __init__(self, retrievers: Sequence[Retriever], *, candidates: int = 20) -> None:
        if len(retrievers) == 0:
            raise ValueError("Retriever list is empty")
        self.retrievers = retrievers
        self.candidates = candidates

    async def search(self, query: str, k: int = 5) -> list[SearchHit]:
        if k < 1:
            raise ValueError(f"Parameter k must be greater or equal than 1, got {k}")
        rankings = []
        for retriever in self.retrievers:
            ranking = await retriever.search(query=query, k=max(self.candidates, k))
            rankings.append(ranking)
        return reciprocal_rank_fusion(rankings=rankings)[:k]
