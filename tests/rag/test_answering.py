import json

import pytest

from docaudit.llm import FakeProvider, LLMError, Usage
from docaudit.rag import (
    AnswerError,
    Citation,
    ClauseAnswer,
    ClauseAnswerer,
    SearchHit,
    build_answer_prompt,
    verify_citations,
)

from .helpers import ScriptedRetriever, make_chunk

QUESTION = "When must we pay Acme?"
TEXT = "Customer shall pay undisputed amounts within 30 days of receipt."
HIT = SearchHit(
    chunk=make_chunk("MSA#4.1", TEXT, doc_title="Agreement with Acme", heading="4. Payment"),
    score=1.0,
)
OTHER = SearchHit(chunk=make_chunk("MSA#9.1", "Termination needs 60 days of notice."), score=0.5)
HITS = [HIT, OTHER]


def reply(quote: str = "within 30 days", chunk_id: str = "MSA#4.1") -> str:
    citation = Citation(chunk_id=chunk_id, quote=quote)
    return ClauseAnswer(
        answerable=True, answer="In 30 days.", citations=[citation]
    ).model_dump_json()


def answer_with(*citations: Citation, answerable: bool = True) -> ClauseAnswer:
    return ClauseAnswer(answerable=answerable, answer="...", citations=list(citations))


# --- build_answer_prompt ---------------------------------------------------------


def test_prompt_separates_instructions_from_data() -> None:
    system, messages = build_answer_prompt(QUESTION, HITS)
    assert [m.role for m in messages] == ["user"]
    content = messages[0].content
    assert TEXT not in system
    assert QUESTION not in system
    assert "<question>" in content
    assert QUESTION in content
    assert '<chunk id="MSA#4.1">' in content
    assert TEXT in content
    assert "Termination needs" in content
    assert content.count("</chunk>") == 2


def test_prompt_gives_the_model_context_and_the_schema() -> None:
    system, messages = build_answer_prompt(QUESTION, HITS)
    assert "Agreement with Acme" in messages[0].content, "whose clause is it?"
    assert "4. Payment" in messages[0].content
    assert "answerable" in system
    assert "citations" in system
    assert json.dumps(ClauseAnswer.model_json_schema()) in system, "real JSON, not a Python dict"


# --- verify_citations ------------------------------------------------------------


@pytest.mark.parametrize(
    "quote",
    ["within 30 days", "WITHIN 30 DAYS", "within   30\ndays"],
    ids=["verbatim", "other-case", "other-whitespace"],
)
def test_valid_quotes(quote: str) -> None:
    assert verify_citations(answer_with(Citation(chunk_id="MSA#4.1", quote=quote)), HITS) == []


def test_not_answerable_with_no_citations_is_valid() -> None:
    assert verify_citations(answer_with(answerable=False), HITS) == []


def test_answerable_needs_a_citation() -> None:
    assert len(verify_citations(answer_with(), HITS)) == 1


def test_not_answerable_must_not_cite() -> None:
    citation = Citation(chunk_id="MSA#4.1", quote="within 30 days")
    assert len(verify_citations(answer_with(citation, answerable=False), HITS)) == 1


def test_unknown_chunk_is_named() -> None:
    problems = verify_citations(answer_with(Citation(chunk_id="MSA#7.7", quote="x")), HITS)
    assert len(problems) == 1
    assert "MSA#7.7" in problems[0]


def test_invented_quote_is_named() -> None:
    bad = Citation(chunk_id="MSA#4.1", quote="within 90 days")
    problems = verify_citations(answer_with(bad), HITS)
    assert len(problems) == 1
    assert "MSA#4.1" in problems[0]


def test_quote_must_be_in_the_chunk_it_cites() -> None:
    wrong_chunk = Citation(chunk_id="MSA#9.1", quote="within 30 days")
    assert len(verify_citations(answer_with(wrong_chunk), HITS)) == 1


def test_every_problem_is_reported() -> None:
    bad = [Citation(chunk_id="X", quote="a"), Citation(chunk_id="MSA#4.1", quote="b")]
    assert len(verify_citations(answer_with(*bad), HITS)) == 2


# --- ClauseAnswerer --------------------------------------------------------------


async def test_answers_with_verified_citations() -> None:
    retriever = ScriptedRetriever(HITS)
    fake = FakeProvider(replies=[reply()])
    result = await ClauseAnswerer(retriever, fake, k=3).answer(QUESTION)

    assert result.answer.citations[0].chunk_id == "MSA#4.1"
    assert result.attempts == 1
    assert result.hits == HITS
    assert retriever.calls == [(QUESTION, 3)]
    call = fake.calls[0]
    assert call.temperature == 0.0
    assert call.response_schema == ClauseAnswer.model_json_schema()


async def test_nothing_retrieved_means_no_model_call() -> None:
    fake = FakeProvider(replies=[])
    result = await ClauseAnswerer(ScriptedRetriever([]), fake).answer(QUESTION)
    assert result.answer.answerable is False
    assert result.attempts == 0
    assert fake.calls == []


async def test_repairs_an_invented_quote_without_retrieving_again() -> None:
    retriever = ScriptedRetriever(HITS)
    fake = FakeProvider(replies=[reply(quote="within 90 days"), reply()])
    result = await ClauseAnswerer(retriever, fake).answer(QUESTION)

    assert result.attempts == 2
    assert len(retriever.calls) == 1, "the evidence must not change between repairs"
    retry = fake.calls[1].messages
    assert [m.role for m in retry] == ["user", "assistant", "user"]
    assert "MSA#4.1" in retry[-1].content


@pytest.mark.parametrize(
    "bad",
    ["not json at all", '{"answerable": true}'],
    ids=["unparseable", "invalid-schema"],
)
async def test_repairs_unusable_output(bad: str) -> None:
    fake = FakeProvider(replies=[bad, reply()])
    result = await ClauseAnswerer(ScriptedRetriever(HITS), fake).answer(QUESTION)
    assert result.attempts == 2


async def test_gives_up_after_max_attempts() -> None:
    fake = FakeProvider(replies=[reply(quote="invented")] * 5)
    answerer = ClauseAnswerer(ScriptedRetriever(HITS), fake, max_attempts=2)
    with pytest.raises(AnswerError) as excinfo:
        await answerer.answer(QUESTION)
    assert len(fake.calls) == 2
    assert "2" in str(excinfo.value)


async def test_chains_the_last_exception_when_there_is_one() -> None:
    fake = FakeProvider(replies=["nope"])
    answerer = ClauseAnswerer(ScriptedRetriever(HITS), fake, max_attempts=1)
    with pytest.raises(AnswerError) as excinfo:
        await answerer.answer(QUESTION)
    assert excinfo.value.__cause__ is not None


async def test_provider_errors_propagate() -> None:
    fake = FakeProvider(replies=[])
    with pytest.raises(LLMError):
        await ClauseAnswerer(ScriptedRetriever(HITS), fake).answer(QUESTION)
    assert len(fake.calls) == 1


async def test_usage_is_summed_and_max_tokens_is_passed() -> None:
    fake = FakeProvider(
        replies=["nope", reply()], usage_per_call=Usage(input_tokens=10, output_tokens=5)
    )
    result = await ClauseAnswerer(ScriptedRetriever(HITS), fake, max_tokens=77).answer(QUESTION)
    assert result.usage == Usage(input_tokens=20, output_tokens=10)
    assert fake.calls[0].max_tokens == 77
    assert not fake.closed


def test_rejects_max_attempts_below_one() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        ClauseAnswerer(ScriptedRetriever([]), FakeProvider(replies=[]), max_attempts=0)


def test_blank_quote_is_rejected() -> None:
    problems = verify_citations(answer_with(Citation(chunk_id="MSA#4.1", quote="   ")), HITS)
    assert len(problems) == 1
    assert "MSA#4.1" in problems[0]


async def test_failure_reports_the_last_error_and_does_not_chain_an_older_one() -> None:
    fake = FakeProvider(replies=["nope", reply(quote="invented")])
    answerer = ClauseAnswerer(ScriptedRetriever(HITS), fake, max_attempts=2)
    with pytest.raises(AnswerError) as excinfo:
        await answerer.answer(QUESTION)
    assert "MSA#4.1" in str(excinfo.value)
    assert excinfo.value.__cause__ is None
