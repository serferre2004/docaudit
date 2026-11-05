import pytest

from docaudit.rag import HybridRetriever, SearchHit, reciprocal_rank_fusion

from .helpers import ScriptedRetriever, hits_for, make_chunk


def test_rrf_score_uses_positions_only() -> None:
    fused = reciprocal_rank_fusion([hits_for("A", "B"), hits_for("B")], constant=60)
    scores = {hit.chunk.id: hit.score for hit in fused}
    assert scores["A"] == pytest.approx(1 / 61)
    assert scores["B"] == pytest.approx(1 / 62 + 1 / 61)
    assert [hit.chunk.id for hit in fused] == ["B", "A"], "agreement between rankings wins"


def test_rrf_ignores_the_original_scales() -> None:
    small = hits_for("A", "B")
    huge = [hit.model_copy(update={"score": hit.score * 1000}) for hit in hits_for("C", "D")]
    fused = reciprocal_rank_fusion([small, huge])
    assert fused[0].score == pytest.approx(fused[1].score), "first places count the same"


def test_rrf_deduplicates_and_ties_keep_first_appearance() -> None:
    fused = reciprocal_rank_fusion([hits_for("A", "B"), hits_for("C", "A")], constant=1)
    ids = [hit.chunk.id for hit in fused]
    assert ids == ["A", "C", "B"]
    assert len(ids) == len(set(ids))


def test_rrf_with_nothing_to_fuse() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


async def test_hybrid_fuses_and_cuts_at_k() -> None:
    hybrid = HybridRetriever(
        [ScriptedRetriever(hits_for("A", "B", "C")), ScriptedRetriever(hits_for("C", "D", "A"))]
    )
    hits = await hybrid.search("q", k=2)
    assert {hit.chunk.id for hit in hits} == {"A", "C"}


async def test_hybrid_asks_for_more_candidates_than_it_returns() -> None:
    first, second = ScriptedRetriever([]), ScriptedRetriever([])
    hybrid = HybridRetriever([first, second], candidates=20)
    await hybrid.search("q", k=5)
    await hybrid.search("q", k=30)
    assert first.calls == [("q", 20), ("q", 30)]
    assert second.calls == [("q", 20), ("q", 30)]


async def test_hybrid_validates_its_arguments() -> None:
    with pytest.raises(ValueError):
        HybridRetriever([])
    with pytest.raises(ValueError):
        await HybridRetriever([ScriptedRetriever([])]).search("q", k=0)


def test_fusion_identifies_chunks_by_id() -> None:
    first = SearchHit(chunk=make_chunk("A", "one version"), score=1.0)
    second = SearchHit(chunk=make_chunk("A", "another version"), score=1.0)
    fused = reciprocal_rank_fusion([[first], [second]])
    assert [hit.chunk.id for hit in fused] == ["A"]
    assert fused[0].score == pytest.approx(2 / 61)
