"""Tracing every model call.   >>> YOUR TURN: implement ``TracingProvider.complete``. <<<

Tests that define "done": tests/observability/test_tracing.py

``TracingProvider`` wraps any ``LLMProvider`` and has the same shape, so nothing else in the code
changes: the graph, the extractor and the tool loop still see "a provider". This is the decorator
pattern, and it is why the ``LLMProvider`` protocol paid off: tracing is added in ONE place.

Which document is a call for? The provider can't know: it only sees messages. The caller says so
once, with ``track_document(document_id)``, and a ``ContextVar`` carries it down to every call made
inside that block, across ``await`` and into the tasks LangGraph starts. No parameter has to be
threaded through the graph, the nodes and the tool loop.
"""

import logging
import time
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from decimal import Decimal
from types import TracebackType
from typing import Any, Self

from pydantic import BaseModel, ConfigDict

from docaudit.llm.base import LLMProvider
from docaudit.llm.types import LLMError, LLMResponse, Message, ToolDefinition, Usage
from docaudit.observability.costs import PRICES, Price, cost_usd

logger = logging.getLogger("docaudit.llm")

current_document: ContextVar[str | None] = ContextVar("current_document", default=None)


@contextmanager
def track_document(document_id: str) -> Iterator[None]:
    """Attribute every model call inside this block to ``document_id`` (provided)."""
    token = current_document.set(document_id)
    try:
        yield
    finally:
        current_document.reset(token)


class LLMCall(BaseModel):
    """One model call, as the trace sees it."""

    model_config = ConfigDict(frozen=True)

    document_id: str | None
    provider: str
    model: str
    usage: Usage
    latency_ms: float
    cost_usd: Decimal | None
    error: str | None = None


class UsageSummary(BaseModel):
    """Every call for one document, added up."""

    model_config = ConfigDict(frozen=True)

    calls: int = 0
    errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    # None as soon as one call has no price: a partial sum would look like a real total.
    cost_usd: Decimal | None = Decimal(0)


class UsageTracker:
    """In-memory store of calls (provided).

    Enough for one process. With several API workers each would see only its own calls: you'd
    send them to a shared store instead (a table, Prometheus, an OpenTelemetry collector).
    """

    def __init__(self) -> None:
        self._calls: dict[str | None, list[LLMCall]] = defaultdict(list)

    def record(self, call: LLMCall) -> None:
        self._calls[call.document_id].append(call)

    def calls_for(self, document_id: str | None) -> list[LLMCall]:
        return list(self._calls.get(document_id, []))

    def summary(self, document_id: str | None) -> UsageSummary:
        calls = self.calls_for(document_id)
        known = [c.cost_usd for c in calls if c.cost_usd is not None]
        return UsageSummary(
            calls=len(calls),
            errors=sum(1 for c in calls if c.error is not None),
            input_tokens=sum(c.usage.input_tokens for c in calls),
            output_tokens=sum(c.usage.output_tokens for c in calls),
            latency_ms=round(sum(c.latency_ms for c in calls), 1),
            cost_usd=sum(known, Decimal(0)) if len(known) == len(calls) else None,
        )


class TracingProvider:
    """An ``LLMProvider`` that records every call it forwards to ``inner``."""

    def __init__(
        self, inner: LLMProvider, tracker: UsageTracker, prices: Mapping[str, Price] = PRICES
    ) -> None:
        self.inner = inner
        self.tracker = tracker
        self.prices = prices
        self.name = inner.name

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 1024,
        response_schema: dict[str, Any] | None = None,
        tools: Sequence[ToolDefinition] | None = None,
    ) -> LLMResponse:
        """Forward the call unchanged, and record it.

        Requirements:
          * Pass every argument through to ``inner.complete`` and return its response untouched.
          * Measure the latency with ``time.perf_counter()`` (wall-clock time can jump), in ms.
          * Record one ``LLMCall`` in the tracker, with the document from ``current_document``,
            the model the RESPONSE reports, its usage, and ``cost_usd(..., self.prices)``.
          * If ``inner`` raises ``LLMError``: record the call anyway (``model`` = ``"unknown"``,
            empty usage, ``cost_usd`` = 0, ``error`` = the message), then re-raise it. Failed
            calls are the ones you most need to see.
          * Log one line per call on ``logger`` (``docaudit.llm``) with ``extra=`` holding the
            call's fields, so a JSON log formatter can index them. Warning level for errors.
        """
        start = time.perf_counter()
        try:
            response = await self.inner.complete(
                messages,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                response_schema=response_schema,
                tools=tools,
            )
        except LLMError as exc:
            call = LLMCall(
                document_id=current_document.get(),
                provider=self.name,
                model="unknown",
                usage=Usage(),
                latency_ms=(time.perf_counter() - start) * 1000,
                cost_usd=Decimal(),
                error=str(exc),
            )
            self.tracker.record(call)
            logger.warning("llm_call_failed", extra=call.model_dump(mode="json"))
            raise
        call = LLMCall(
            document_id=current_document.get(),
            provider=self.name,
            model=response.model,
            usage=response.usage,
            latency_ms=(time.perf_counter() - start) * 1000,
            cost_usd=cost_usd(response.model, response.usage, self.prices),
        )
        self.tracker.record(call)
        logger.info("llm_call", extra=call.model_dump(mode="json"))
        return response

    async def aclose(self) -> None:
        await self.inner.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()
