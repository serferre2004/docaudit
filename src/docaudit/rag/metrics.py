"""Retrieval metrics.   >>> YOUR TURN: implement this file. <<<

Tests that define "done": tests/rag/test_metrics.py

If the right clause is never retrieved, no prompt can save the answer. These metrics measure
retrieval alone, with no LLM involved: they are fast, free and deterministic.
"""

from collections.abc import Sequence

from docaudit.rag.schemas import EvalQuestion, SearchHit

RetrievalResult = tuple[EvalQuestion, Sequence[SearchHit]]


def reciprocal_rank(question: EvalQuestion, hits: Sequence[SearchHit]) -> float:
    """Return ``1 / position`` of the first correct hit, or ``0.0`` if there is none.

    A hit is correct when its chunk has the question's ``doc_id`` AND ``section``. Compare those,
    not chunk ids: any part of a long clause (``#8.1-p2``) counts as finding clause 8.1.
    """
    for pos, hit in enumerate(hits):
        chunk = hit.chunk
        if chunk.doc_id == question.doc_id and chunk.section == question.section:
            return 1 / (pos + 1)
    return 0.0


def hit_rate(results: Sequence[RetrievalResult], k: int) -> float:
    """Return the fraction of questions with a correct hit among their first ``k`` hits.

    Empty ``results`` or ``k < 1`` raise ``ValueError``.
    """
    if len(results) == 0:
        raise ValueError("results list is empty")
    if k < 1:
        raise ValueError(f"Parameter k must be greater or equal than 1, got {k}")
    threshold = 1 / k
    total = 0
    for result in results:
        if reciprocal_rank(result[0], result[1]) >= threshold:
            total += 1
    return total / len(results)


def mean_reciprocal_rank(results: Sequence[RetrievalResult]) -> float:
    """Return the average ``reciprocal_rank``. Empty ``results`` raise ``ValueError``.

    Hit rate only asks "is it there?". This also rewards putting it first.
    """
    if len(results) == 0:
        raise ValueError("results list is empty")
    total = 0.0
    for result in results:
        total += reciprocal_rank(result[0], result[1])
    return total / len(results)
