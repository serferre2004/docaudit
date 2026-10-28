"""Anthropic (Claude) backend.   >>> YOUR TURN: implement this file. <<<

Docs: https://docs.claude.com/en/api/messages  (Messages API)
Tests that define "done": tests/llm/test_providers.py  (the ``anthropic`` ones)
"""

from collections.abc import Sequence
from typing import Any

from docaudit.llm.types import LLMResponse, Message


def build_payload(
    messages: Sequence[Message],
    *,
    model: str,
    system: str | None,
    temperature: float | None,
    max_tokens: int,
) -> dict[str, Any]:
    """Translate our neutral types into the kwargs for ``client.messages.create``.

    Requirements:
      * Always include ``model``, ``max_tokens`` and ``messages``
        (each message becomes ``{"role": ..., "content": ...}``).
      * Include ``system`` ONLY if a non-empty system prompt was given
        (it is a top-level field, not a message).
      * Include ``temperature`` ONLY if it is not ``None``.

    Keep this a pure function: no network, no client. That is what makes it trivially testable.
    """
    raise NotImplementedError


class AnthropicProvider:
    """Implements the ``LLMProvider`` protocol (see ``docaudit/llm/base.py``).

    Requirements:
      * ``name`` class attribute equal to ``"anthropic"``.
      * ``__init__(self, *, api_key: str, model: str, timeout_s: float = 120.0)`` creates an
        ``anthropic.AsyncAnthropic`` client (look at its ``timeout`` and ``max_retries`` args).
      * ``complete`` builds the payload, awaits ``client.messages.create(**payload)`` and returns
        an ``LLMResponse``.
          - ``resp.content`` is a LIST OF BLOCKS, not a string. Join the text blocks.
          - Map ``resp.usage`` to our ``Usage`` and ``resp.stop_reason`` to ``stop_reason``.
      * Any ``anthropic.APIError`` must be re-raised as our ``LLMError`` (keep the cause with
        ``raise ... from exc``) so no caller ever depends on a vendor exception.
    """

    name = "anthropic"

    def __init__(self, *, api_key: str, model: str, timeout_s: float = 120.0) -> None:
        raise NotImplementedError

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        raise NotImplementedError
