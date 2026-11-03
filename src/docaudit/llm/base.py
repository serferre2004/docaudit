from collections.abc import Sequence
from types import TracebackType
from typing import Any, Protocol, Self

from docaudit.llm.types import LLMResponse, Message


class LLMProvider(Protocol):
    """The contract every backend implements.

    A ``Protocol`` (structural typing) means providers don't need to inherit from
    anything: any class with this shape works. That keeps tests and fakes trivial.

    Providers own network resources, so they are async context managers: use
    ``async with create_provider(settings) as provider:`` and they close themselves
    when the block exits, even if it exits with an exception.
    """

    name: str

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 1024,
        response_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Return one assistant reply for the conversation.

        ``temperature=None`` means "use the provider's default" — some newer models
        restrict sampling parameters, so we only send it when explicitly asked.

        ``response_schema`` is an optional hint: a JSON schema the reply should follow.
        Backends with native support enforce it, the others may ignore it, so it is never a
        guarantee. Callers must still validate the reply.
        """
        ...

    async def aclose(self) -> None:
        """Release network resources (connection pools). The provider is unusable afterwards."""
        ...

    async def __aenter__(self) -> Self: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Must call ``aclose()``."""
        ...
