"""Ollama (local models) backend.   >>> YOUR TURN: implement this file. <<<

Docs: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion
Tests that define "done": tests/llm/test_providers.py  (the ``ollama`` ones)

We deliberately use ``httpx`` directly instead of a client library: an LLM API is just a JSON
POST, and ``httpx.MockTransport`` lets us test the provider with no server at all.
"""

from collections.abc import Sequence
from types import TracebackType
from typing import Any, Self

import httpx

from docaudit.llm.types import LLMError, LLMResponse, Message, ToolCall, ToolDefinition, Usage


def tools_to_wire(tools: Sequence[ToolDefinition]) -> list[dict[str, Any]]:
    """Translate tool definitions for the ``tools`` field.   >>> YOUR TURN <<<

    Each one becomes
    ``{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}``:
    same information as Anthropic, different shape.
    """
    wired_tools = []
    for tool in tools:
        function_field = {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        }
        wired_tools.append({"type": "function", "function": function_field})
    return wired_tools


def message_to_wire(message: Message) -> list[dict[str, Any]]:
    """Translate one message into one or MORE wire messages.   >>> YOUR TURN <<<

    Requirements:
      * A plain message becomes ``[{"role": ..., "content": "text"}]``.
      * An ``assistant`` message with ``tool_calls`` keeps its text in ``content`` and adds
        ``"tool_calls": [{"function": {"name": ..., "arguments": ...}}, ...]``.
      * Each tool result becomes its OWN message:
        ``{"role": "tool", "tool_name": ..., "content": ...}``. Ollama has no error flag, so when
        ``is_error`` is true prefix the content with ``"ERROR: "``: it's the only way the model
        finds out.
    """
    wire_messages: list[dict[str, Any]] = []
    if not (message.tool_results or message.tool_calls):
        wire_messages = [{"role": message.role, "content": message.content}]
    elif message.role == "assistant" and message.tool_calls:
        tool_calls = [
            {"function": {"name": call.name, "arguments": call.arguments}}
            for call in message.tool_calls
        ]
        wire_messages = [
            {"role": message.role, "content": message.content, "tool_calls": tool_calls}
        ]
    elif message.tool_results:
        for result in message.tool_results:
            content = result.content
            if result.is_error:
                content = "ERROR: " + content
            wire_messages.append({"role": "tool", "tool_name": result.name, "content": content})
    return wire_messages


def parse_tool_calls(message: dict[str, Any]) -> tuple[ToolCall, ...]:
    """Turn ``data["message"]`` into ``ToolCall`` s.   >>> YOUR TURN <<<

    ``tool_calls`` may be missing. Each one looks like
    ``{"function": {"name": ..., "arguments": {...}}}``. Ollama doesn't require call ids, so use
    the ``id`` if there is one and ``f"call_{index}"`` otherwise: our types need one.
    """
    return tuple(
        ToolCall(
            id=call.get("id", f"call_{index}"),
            name=call["function"]["name"],
            arguments=call["function"]["arguments"],
        )
        for index, call in enumerate(message.get("tool_calls") or [])
    )


def build_payload(
    messages: Sequence[Message],
    *,
    model: str,
    system: str | None,
    temperature: float | None,
    max_tokens: int,
    response_schema: dict[str, Any] | None = None,
    tools: Sequence[ToolDefinition] | None = None,
) -> dict[str, Any]:
    """Translate our neutral types into the JSON body for ``POST /api/chat``.

    Requirements (note how they differ from Anthropic!):
      * The system prompt is NOT a separate field: it becomes the FIRST message with
        ``role: "system"`` (only if one was given).
      * ``stream`` must be ``False`` so we get a single JSON response.
      * Sampling params go under ``options``: ``num_predict`` is Ollama's name for max tokens.
        Include ``temperature`` in ``options`` ONLY if it is not ``None``.
      * ``response_schema`` goes in the top-level ``format`` field, ONLY if it is not ``None``.
      * >>> YOUR TURN (Lesson 4) <<< Build the messages with ``message_to_wire``, and include
        ``tools`` ONLY if some were given (``tools_to_wire``).
    """
    msg_list = []
    if system:
        msg_list.append({"role": "system", "content": system})
    for msg in messages:
        msg_list.extend(message_to_wire(msg))
    options: dict[str, object] = {"num_predict": max_tokens}
    if temperature is not None:
        options["temperature"] = temperature
    base_dict: dict[str, Any] = {
        "model": model,
        "stream": False,
        "options": options,
        "messages": msg_list,
    }
    if response_schema is not None:
        base_dict["format"] = response_schema
    if tools:
        base_dict["tools"] = tools_to_wire(tools)
    return base_dict


class OllamaProvider:
    """Implements the ``LLMProvider`` protocol.

    Requirements:
      * ``name`` class attribute equal to ``"ollama"``.
      * ``__init__(self, *, base_url, model, timeout_s=120.0, transport=None)`` creates an
        ``httpx.AsyncClient(base_url=..., timeout=..., transport=...)``. The ``transport``
        argument exists purely so tests can inject ``httpx.MockTransport``.
      * ``complete`` POSTs the payload to ``/api/chat`` and returns an ``LLMResponse``.
          - Response text:   ``data["message"]["content"]``
          - Token counts:    ``prompt_eval_count`` / ``eval_count`` (may be missing -> 0)
          - Stop reason:     ``done_reason``
      * Any ``httpx.HTTPError`` (including non-2xx via ``raise_for_status``) becomes ``LLMError``.
      * ``aclose()`` closes the client. Think: who is responsible for calling it?
      * >>> YOUR TURN (Lesson 4) <<< Pass ``tools`` to ``build_payload`` and fill
        ``LLMResponse.tool_calls`` with ``parse_tool_calls``.
    """

    name = "ollama"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_s: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        self.client = httpx.AsyncClient(base_url=base_url, transport=transport, timeout=timeout_s)

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
        try:
            response = await self.client.post(
                "/api/chat",
                json=build_payload(
                    messages,
                    model=self.model,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_schema=response_schema,
                    tools=tools,
                ),
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc
        res = response.json()
        usage = Usage(
            input_tokens=res.get("prompt_eval_count", 0),
            output_tokens=res.get("eval_count", 0),
        )
        return LLMResponse(
            model=res["model"],
            text=res["message"]["content"],
            stop_reason=res.get("done_reason"),
            usage=usage,
            tool_calls=parse_tool_calls(res["message"]),
        )

    async def aclose(self) -> None:
        await self.client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()
