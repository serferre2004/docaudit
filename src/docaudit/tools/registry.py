"""The boundary between what the model asks for and what actually runs.   >>> YOUR TURN <<<

Tests that define "done": tests/tools/test_registry.py

A tool call is untrusted input. The model can invent a tool, leave out an argument or send the
wrong type, so nothing reaches a handler until it has been checked here.
"""

import json
from collections.abc import Sequence

from pydantic import ValidationError

from docaudit.extraction.prompts import format_validation_error
from docaudit.llm.types import ToolCall, ToolDefinition, ToolResult
from docaudit.tools.schemas import Tool, ToolError


class ToolRegistry:
    """Holds the available tools and runs them on request.

    Requirements:
      * ``__init__(self, tools)``: two tools with the same name raise ``ValueError``.
      * ``definitions`` (a property): one ``ToolDefinition`` per tool, in order, with
        ``input_schema=tool.input_model.model_json_schema()``.
      * ``execute(call)`` returns a ``ToolResult`` with the call's ``id`` and ``name``:
          - Success: ``content`` is the handler's dict as JSON, using ``default=str`` so a
            ``Decimal`` doesn't crash ``json.dumps``.
          - Unknown tool: ``is_error=True``, and the content lists the available tool names.
          - Arguments that fail ``input_model.model_validate``: ``is_error=True``, with the content
            from ``format_validation_error`` (Lesson 2). The handler must NOT run.
          - The handler raises ``ToolError``: ``is_error=True``, with the error's message.
      * Any other exception from a handler propagates. A bug in our code is not the model's
        mistake, and sending it back would hide the bug and leak internals to the model.

    Why return errors instead of raising? The model can read an error and correct itself on the
    next turn, the same idea as the repair loop in Lesson 2.
    """

    def __init__(self, tools: Sequence[Tool]) -> None:
        self.tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self.tools:
                raise ValueError(f"There is more than one tool named {tool.name}")
            self.tools[tool.name] = tool

    @property
    def definitions(self) -> list[ToolDefinition]:
        tool_defs = []
        for name, tool in self.tools.items():
            definition = ToolDefinition(
                name=name,
                description=tool.description,
                input_schema=tool.input_model.model_json_schema(),
            )
            tool_defs.append(definition)
        return tool_defs

    def execute(self, call: ToolCall) -> ToolResult:
        tool = self.tools.get(call.name, None)
        if tool is None:
            names = ", ".join(self.tools)
            content = f"Unknown tool {call.name!r}. Available tools: {names}"
            return ToolResult(tool_call_id=call.id, name=call.name, content=content, is_error=True)
        # Two separate try blocks: only the model's arguments are validated here. A
        # ValidationError raised INSIDE a handler is our bug and must propagate.
        try:
            args_obj = tool.input_model.model_validate(call.arguments)
        except ValidationError as exc:
            content = format_validation_error(exc)
            return ToolResult(tool_call_id=call.id, name=call.name, content=content, is_error=True)
        try:
            result = tool.handler(args_obj)
        except ToolError as exc:
            return ToolResult(tool_call_id=call.id, name=call.name, content=str(exc), is_error=True)
        content = json.dumps(result, default=str)
        return ToolResult(tool_call_id=call.id, name=call.name, content=content)
