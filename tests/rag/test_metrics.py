import pytest

from docaudit.rag import (
    EvalQuestion,
    SearchHit,
    hit_rate,
    mean_reciprocal_rank,
    reciprocal_rank,
)

from .helpers import make_chunk


def question(doc_id: str = "MSA", section: str = "4.1") -> EvalQuestion:
    return EvalQuestion(id="q", question="?", doc_id=doc_id, section=section, kind="paraphrase")


def hit(doc_id: str, section: str, part: str = "") -> SearchHit:
    chunk = make_chunk(f"{doc_id}#{section}{part}", doc_id=doc_id, section=section)
    return SearchHit(chunk=chunk, score=1.0)


RIGHT = hit("MSA", "4.1")
WRONG = hit("MSA", "9.9")
SAME_SECTION_OTHER_DOC = hit("OTHER", "4.1")


def test_reciprocal_rank_uses_the_first_correct_position() -> None:
    assert reciprocal_rank(question(), [RIGHT, WRONG]) == 1.0
    assert reciprocal_rank(question(), [WRONG, RIGHT, RIGHT]) == 0.5
    assert reciprocal_rank(question(), [WRONG]) == 0.0
    assert reciprocal_rank(question(), []) == 0.0


def test_needs_both_document_and_section() -> None:
    assert reciprocal_rank(question(), [SAME_SECTION_OTHER_DOC]) == 0.0


def test_any_part_of_a_long_clause_counts() -> None:
    assert reciprocal_rank(question(), [hit("MSA", "4.1", "-p2")]) == 1.0


def test_hit_rate_respects_k() -> None:
    results = [(question(), [WRONG, RIGHT]), (question(), [RIGHT]), (question(), [WRONG])]
    assert hit_rate(results, k=1) == pytest.approx(1 / 3)
    assert hit_rate(results, k=2) == pytest.approx(2 / 3)


def test_mean_reciprocal_rank() -> None:
    results = [(question(), [RIGHT]), (question(), [WRONG, RIGHT]), (question(), [WRONG])]
    assert mean_reciprocal_rank(results) == pytest.approx((1 + 0.5 + 0) / 3)


def test_metrics_reject_meaningless_input() -> None:
    with pytest.raises(ValueError):
        hit_rate([], k=1)
    with pytest.raises(ValueError):
        hit_rate([(question(), [RIGHT])], k=0)
    with pytest.raises(ValueError):
        mean_reciprocal_rank([])
