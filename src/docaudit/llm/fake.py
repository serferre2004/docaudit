from collections.abc import Sequence
from dataclasses import dataclass, field
from types import TracebackType
from typing import Self

from docaudit.llm.types import LLMError, LLMResponse, Message, Usage


@dataclass
class RecordedCall:
    messages: list[Message]
    system: str | None
    temperature: float | None
    max_tokens: int


@dataclass
class FakeProvider:
    """A scripted provider for deterministic tests — no network, no cost, no flakiness.

    Give it the replies it should return, in order. It records every call so tests
    can assert on exactly what the code under test sent to "the model". Every reply
    reports ``usage_per_call``, so tests can check token accounting.
    """

    replies: list[str]
    name: str = "fake"
    calls: list[RecordedCall] = field(default_factory=list)
    closed: bool = False
    usage_per_call: Usage = field(default_factory=Usage)

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        if self.closed:
            raise LLMError("FakeProvider was used after aclose()")
        self.calls.append(RecordedCall(list(messages), system, temperature, max_tokens))
        if not self.replies:
            raise LLMError("FakeProvider ran out of scripted replies")
        return LLMResponse(
            text=self.replies.pop(0),
            model="fake-model",
            stop_reason="end_turn",
            usage=self.usage_per_call,
        )

    async def aclose(self) -> None:
        self.closed = True

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()
