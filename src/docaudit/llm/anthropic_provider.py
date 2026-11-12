"""Anthropic (Claude) backend.   >>> YOUR TURN: implement this file. <<<

Docs: https://docs.claude.com/en/api/messages  (Messages API)
Tests that define "done": tests/llm/test_providers.py  (the ``anthropic`` ones)
"""

from collections.abc import Sequence
from types import TracebackType
from typing import Any, Self

import httpx2
from anthropic import APIError, AsyncAnthropic

from docaudit.llm.types import LLMError, LLMResponse, Message, ToolCall, ToolDefinition, Usage


def tools_to_wire(tools: Sequence[ToolDefinition]) -> list[dict[str, Any]]:
    """Translate tool definitions for the ``tools`` field.   >>> YOUR TURN <<<

    Each one becomes ``{"name": ..., "description": ..., "input_schema": ...}``.
    """
    return [
        {"name": tool.name, "description": tool.description, "input_schema": tool.input_schema}
        for tool in tools
    ]


def message_to_wire(message: Message) -> dict[str, Any]:
    """Translate one message.   >>> YOUR TURN <<<

    Requirements:
      * A plain message stays ``{"role": ..., "content": "text"}``.
      * An ``assistant`` message with ``tool_calls`` has a LIST of blocks as its ``content``: a
        ``{"type": "text", "text": ...}`` block ONLY if the text isn't empty (the API rejects empty
        text blocks), then one ``{"type": "tool_use", "id": ..., "name": ..., "input": ...}`` per
        call.
      * A ``user`` message with ``tool_results`` has one block per result:
        ``{"type": "tool_result", "tool_use_id": ..., "content": ..., "is_error": ...}``.
    """
    wire_messages: dict[str, Any]
    content: list[dict[str, Any]]
    if not (message.tool_results or message.tool_calls):
        wire_messages = {"role": message.role, "content": message.content}
    elif message.role == "assistant" and message.tool_calls:
        content = [
            {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
            for call in message.tool_calls
        ]
        if message.content:
            content.insert(0, {"type": "text", "text": message.content})
        wire_messages = {"role": message.role, "content": content}
    elif message.tool_results:
        content = [
            {
                "type": "tool_result",
                "tool_use_id": result.tool_call_id,
                "content": result.content,
                "is_error": result.is_error,
            }
            for result in message.tool_results
        ]
        wire_messages = {"role": message.role, "content": content}
    else:
        # Only reachable with tool_calls on a non-assistant message: say so instead of failing
        # with an UnboundLocalError.
        raise ValueError(f"A {message.role!r} message cannot carry tool_calls")
    return wire_messages


def parse_tool_calls(content: Sequence[Any]) -> tuple[ToolCall, ...]:
    """Turn the ``tool_use`` blocks of ``resp.content`` into ``ToolCall`` s.   >>> YOUR TURN <<<

    A block has ``.type``, ``.id``, ``.name`` and ``.input``. Other block types are skipped.
    """
    if not content:
        return tuple()
    return tuple(
        ToolCall(id=block.id, name=block.name, arguments=block.input)
        for block in content
        if block.type == "tool_use"
    )


def build_payload(
    messages: Sequence[Message],
    *,
    model: str,
    system: str | None,
    temperature: float | None,
    max_tokens: int,
    tools: Sequence[ToolDefinition] | None = None,
) -> dict[str, Any]:
    """Translate our neutral types into the kwargs for ``client.messages.create``.

    Requirements:
      * Always include ``model``, ``max_tokens`` and ``messages``
        (each message becomes ``{"role": ..., "content": ...}``).
      * Include ``system`` ONLY if a non-empty system prompt was given
        (it is a top-level field, not a message).
      * Include ``temperature`` ONLY if it is not ``None``.
      * >>> YOUR TURN (Lesson 4) <<< Build ``messages`` with ``message_to_wire``, and include
        ``tools`` ONLY if some were given (``tools_to_wire``).

    Keep this a pure function: no network, no client. That is what makes it trivially testable.
    """
    base_dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [message_to_wire(msg) for msg in messages],
    }
    if system:
        base_dict["system"] = system
    if temperature is not None:
        base_dict["temperature"] = temperature
    if tools:
        base_dict["tools"] = tools_to_wire(tools)
    return base_dict


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
      * ``response_schema`` is accepted to honor the protocol but ignored: this backend relies on
        the schema in the prompt and on the caller's validation.
      * >>> YOUR TURN (Lesson 4) <<< Pass ``tools`` to ``build_payload`` and fill
        ``LLMResponse.tool_calls`` with ``parse_tool_calls``.
    """

    name = "anthropic"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_s: float = 120.0,
        transport: httpx2.AsyncBaseTransport | None = None,
        max_retries: int = 2,
    ) -> None:
        self.model = model
        http_client = httpx2.AsyncClient(transport=transport) if transport is not None else None
        self.client = AsyncAnthropic(
            api_key=api_key, timeout=timeout_s, http_client=http_client, max_retries=max_retries
        )

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
        payload = build_payload(
            messages,
            model=self.model,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
        )
        try:
            message = await self.client.messages.create(
                **payload,
            )
            usage = Usage(
                input_tokens=message.usage.input_tokens, output_tokens=message.usage.output_tokens
            )
            text = "".join([c.text for c in message.content if c.type == "text"])
            return LLMResponse(
                model=message.model,
                stop_reason=message.stop_reason,
                text=text,
                usage=usage,
                tool_calls=parse_tool_calls(message.content),
            )
        except APIError as exc:
            raise LLMError("Error while receiving response from Anthropic") from exc

    async def aclose(self) -> None:
        await self.client.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()
