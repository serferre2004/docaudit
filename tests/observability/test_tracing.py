import logging
from decimal import Decimal

import pytest

from docaudit.llm import FakeProvider, LLMError, LLMResponse, Message, Usage
from docaudit.observability import Price, TracingProvider, UsageTracker, track_document

PRICES = {"fake-model": Price(input_per_mtok=Decimal(1), output_per_mtok=Decimal(2))}
HI = [Message(role="user", content="hi")]


def traced(*replies: str | LLMResponse) -> tuple[TracingProvider, FakeProvider, UsageTracker]:
    fake = FakeProvider(
        replies=list(replies), usage_per_call=Usage(input_tokens=100, output_tokens=10)
    )
    tracker = UsageTracker()
    return TracingProvider(fake, tracker, PRICES), fake, tracker


async def test_the_call_goes_through_unchanged() -> None:
    provider, fake, _ = traced("hello")
    response = await provider.complete(HI, system="be brief", temperature=0.0, max_tokens=7)
    assert response.text == "hello"
    sent = fake.calls[0]
    assert (sent.system, sent.temperature, sent.max_tokens) == ("be brief", 0.0, 7)
    assert sent.messages == HI


async def test_each_call_is_recorded_for_the_current_document() -> None:
    provider, _, tracker = traced("a", "b")
    with track_document("doc-1"):
        await provider.complete(HI)
        await provider.complete(HI)
    calls = tracker.calls_for("doc-1")
    assert len(calls) == 2
    first = calls[0]
    assert (first.document_id, first.provider, first.model) == ("doc-1", "fake", "fake-model")
    assert first.usage == Usage(input_tokens=100, output_tokens=10)
    assert first.cost_usd == Decimal("0.00012")  # 100 * 1/1M + 10 * 2/1M
    assert first.latency_ms >= 0
    assert first.error is None


async def test_a_call_outside_any_document_is_recorded_under_none() -> None:
    provider, _, tracker = traced("a")
    await provider.complete(HI)
    assert len(tracker.calls_for(None)) == 1


async def test_a_failed_call_is_recorded_and_the_error_still_raised() -> None:
    provider, _, tracker = traced()  # no replies: the fake raises LLMError
    with track_document("doc-1"), pytest.raises(LLMError):
        await provider.complete(HI)
    (call,) = tracker.calls_for("doc-1")
    assert call.error is not None and "ran out" in call.error
    assert call.usage == Usage()
    assert tracker.summary("doc-1").errors == 1


async def test_every_call_is_logged_with_its_fields(caplog: pytest.LogCaptureFixture) -> None:
    provider, _, _ = traced("a")
    with caplog.at_level(logging.INFO, logger="docaudit.llm"), track_document("doc-1"):
        await provider.complete(HI)
    (record,) = [r for r in caplog.records if r.name == "docaudit.llm"]
    assert record.levelno == logging.INFO
    assert record.document_id == "doc-1"  # type: ignore[attr-defined]
    assert record.model == "fake-model"  # type: ignore[attr-defined]


async def test_a_failed_call_is_logged_as_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    provider, _, _ = traced()
    with caplog.at_level(logging.INFO, logger="docaudit.llm"), pytest.raises(LLMError):
        await provider.complete(HI)
    assert [r.levelno for r in caplog.records if r.name == "docaudit.llm"] == [logging.WARNING]


async def test_the_summary_adds_up_and_refuses_a_partial_cost() -> None:
    provider, fake, tracker = traced("a", LLMResponse(text="b", model="unpriced"))
    with track_document("doc-1"):
        await provider.complete(HI)
        summary = tracker.summary("doc-1")
        assert summary.cost_usd == Decimal("0.00012")
        await provider.complete(HI)
    summary = tracker.summary("doc-1")
    assert (summary.calls, summary.input_tokens, summary.output_tokens) == (2, 200, 20)
    assert summary.cost_usd is None


async def test_closing_the_tracer_closes_the_provider() -> None:
    provider, fake, _ = traced()
    async with provider:
        pass
    assert fake.closed
