import math

import pytest

from docaudit.rag import BM25Index, Chunk, tokenize

from .helpers import make_chunk


def chunk(chunk_id: str, text: str) -> Chunk:
    return make_chunk(chunk_id, text, doc_title="", heading="")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Clause 4.2, form AP-17!", ["clause", "4.2", "form", "ap-17"]),
        ("Ends here.", ["ends", "here"]),
        ("Año 2026 (renovación)", ["año", "2026", "renovación"]),
        ("a - b . c", ["a", "b", "c"]),
        ("", []),
        ("snake_case", ["snake", "case"]),
    ],
    ids=["identifiers", "trailing-dot", "accents", "lone-separators", "empty", "underscore"],
)
def test_tokenize(text: str, expected: list[str]) -> None:
    assert tokenize(text) == expected


async def test_score_follows_the_formula() -> None:
    index = BM25Index()
    index.add([chunk("A", "payment terms")])
    hit = (await index.search("payment"))[0]
    # N=1, n=1: idf=ln(1 + 0.5/1.5). tf=1 and length=avg_length, so the rest of the formula is 1.
    assert hit.score == pytest.approx(math.log(1 + 0.5 / 1.5))


async def test_rare_terms_weigh_more_than_common_ones() -> None:
    index = BM25Index()
    index.add(
        [
            chunk("COMMON", "the invoice is paid"),
            chunk("RARE", "the form ap-17 is signed"),
            chunk("OTHER", "the invoice is late"),
        ]
    )
    hits = await index.search("invoice ap-17")
    assert hits[0].chunk.id == "RARE"


async def test_repeating_a_term_helps_but_with_diminishing_returns() -> None:
    index = BM25Index(b=0.0)
    index.add(
        [chunk("ONE", "fee"), chunk("TWO", "fee fee"), chunk("TEN", "fee " * 10), chunk("X", "y")]
    )
    scores = {h.chunk.id: h.score for h in await index.search("fee", k=3)}
    assert scores["ONE"] < scores["TWO"] < scores["TEN"]
    assert scores["TEN"] < 5 * scores["TWO"], "tf saturates: k1 is what caps it"


async def test_shorter_chunks_win_when_b_is_on() -> None:
    index = BM25Index()
    index.add([chunk("LONG", "fee " + "filler " * 30), chunk("SHORT", "fee only")])
    assert [h.chunk.id for h in await index.search("fee")] == ["SHORT", "LONG"]


async def test_chunks_with_no_match_are_not_returned() -> None:
    index = BM25Index()
    index.add([chunk("A", "payment terms"), chunk("B", "liability cap")])
    assert [h.chunk.id for h in await index.search("payment")] == ["A"]
    assert await index.search("unrelated") == []


async def test_indexes_search_text_not_just_the_clause() -> None:
    index = BM25Index()
    index.add([make_chunk("A", "pay in 30 days", doc_title="Aston agreement", heading="Payment")])
    assert [h.chunk.id for h in await index.search("aston")] == ["A"]


async def test_repeated_query_terms_count_once() -> None:
    index = BM25Index()
    index.add([chunk("A", "fee"), chunk("B", "other")])
    once = await index.search("fee")
    twice = await index.search("fee fee fee")
    assert once[0].score == pytest.approx(twice[0].score)


async def test_add_updates_statistics_and_cuts_at_k() -> None:
    index = BM25Index()
    index.add([chunk("A", "fee")])
    before = (await index.search("fee"))[0].score
    index.add([chunk("B", "fee"), chunk("C", "fee")])
    hits = await index.search("fee", k=2)
    assert len(hits) == 2
    assert hits[0].score != pytest.approx(before), "idf changes when the term gets more common"


async def test_ties_keep_insertion_order_and_k_below_one_is_rejected() -> None:
    index = BM25Index()
    index.add([chunk("A", "fee"), chunk("B", "fee")])
    assert [h.chunk.id for h in await index.search("fee")] == ["A", "B"]
    with pytest.raises(ValueError):
        await index.search("fee", k=0)
