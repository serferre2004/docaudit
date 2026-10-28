from collections.abc import Sequence
from typing import Protocol

from docaudit.llm.types import LLMResponse, Message


class LLMProvider(Protocol):
    """The contract every backend implements.

    A ``Protocol`` (structural typing) means providers don't need to inherit from
    anything: any class with this shape works. That keeps tests and fakes trivial.
    """

    name: str

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Return one assistant reply for the conversation.

        ``temperature=None`` means "use the provider's default" — some newer models
        restrict sampling parameters, so we only send it when explicitly asked.
        """
        ...
