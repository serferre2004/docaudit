import pytest

from docaudit.rag import VectorIndex, cosine_similarity

from .helpers import ScriptedEmbedder, make_chunk


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([1.0, 0.0], [1.0, 0.0], 1.0),
        ([1.0, 0.0], [0.0, 1.0], 0.0),
        ([1.0, 0.0], [-1.0, 0.0], -1.0),
        ([3.0, 4.0], [6.0, 8.0], 1.0),
        ([1.0, 1.0], [1.0, 0.0], 0.7071067),
        ([0.0, 0.0], [1.0, 0.0], 0.0),
    ],
    ids=["same", "orthogonal", "opposite", "ignores-length", "45-degrees", "zero-vector"],
)
def test_cosine_similarity(a: list[float], b: list[float], expected: float) -> None:
    assert cosine_similarity(a, b) == pytest.approx(expected, abs=1e-6)


def test_cosine_rejects_different_dimensions() -> None:
    with pytest.raises(ValueError):
        cosine_similarity([1.0], [1.0, 0.0])


A = make_chunk("A", "alpha")
B = make_chunk("B", "beta")
C = make_chunk("C", "gamma")
VECTORS = {
    A.search_text: [1.0, 0.0],
    B.search_text: [0.6, 0.8],
    C.search_text: [0.0, 1.0],
    "q-near-b": [0.5, 0.9],
    "q-near-a": [1.0, 0.1],
}


async def test_ranks_by_similarity_and_cuts_at_k() -> None:
    index = VectorIndex(ScriptedEmbedder(VECTORS))
    await index.add([A, B, C])
    hits = await index.search("q-near-b", k=2)
    assert [h.chunk.id for h in hits] == ["B", "C"]
    assert hits[0].score > hits[1].score
    assert hits[0].score == pytest.approx(cosine_similarity(VECTORS["q-near-b"], [0.6, 0.8]))


async def test_embeds_search_text_in_one_call() -> None:
    embedder = ScriptedEmbedder(VECTORS)
    await VectorIndex(embedder).add([A, B])
    assert embedder.calls == [[A.search_text, B.search_text]], "title and heading must be embedded"


async def test_add_accumulates() -> None:
    index = VectorIndex(ScriptedEmbedder(VECTORS))
    await index.add([A])
    await index.add([B, C])
    assert len(await index.search("q-near-a", k=10)) == 3


async def test_ties_keep_insertion_order() -> None:
    same = make_chunk("SAME", "same")
    vectors = {A.search_text: [1.0, 0.0], same.search_text: [1.0, 0.0], "q": [1.0, 0.0]}
    index = VectorIndex(ScriptedEmbedder(vectors))
    await index.add([same, A])
    assert [h.chunk.id for h in await index.search("q")] == ["SAME", "A"]


async def test_does_not_call_the_embedder_needlessly() -> None:
    embedder = ScriptedEmbedder({})
    index = VectorIndex(embedder)
    await index.add([])
    assert await index.search("anything") == []
    assert embedder.calls == []


async def test_rejects_k_below_one() -> None:
    with pytest.raises(ValueError):
        await VectorIndex(ScriptedEmbedder({})).search("q", k=0)
