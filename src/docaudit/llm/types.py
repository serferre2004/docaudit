"""Provider-neutral data types.

These are the only LLM-related types the rest of the codebase should know about.
Each provider translates to and from its own wire format.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

Role = Literal["user", "assistant"]


class ToolDefinition(BaseModel):
    """A tool the model may ask us to run. The model only ever sees this description."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    input_schema: dict[str, Any]


class ToolCall(BaseModel):
    """The model asking us to run a tool. It runs nothing itself: this is only a request."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    arguments: dict[str, Any]


class ToolResult(BaseModel):
    """What running a tool produced, on its way back to the model."""

    model_config = ConfigDict(frozen=True)

    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


class Message(BaseModel):
    """One turn of a conversation.

    An ``assistant`` turn can carry ``tool_calls``, and the ``user`` turn after it carries the
    matching ``tool_results``. Plain conversations never touch either field.

    The system prompt is *not* a message: providers disagree on where it goes
    (Anthropic has a top-level field, Ollama uses a ``system`` role), so it is
    passed separately and each provider places it correctly.
    """

    model_config = ConfigDict(frozen=True)

    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()


class Usage(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: int = 0
    output_tokens: int = 0


class LLMResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    model: str
    stop_reason: str | None = None
    usage: Usage = Usage()
    tool_calls: tuple[ToolCall, ...] = ()


class LLMError(Exception):
    """Raised for any provider failure, so callers never depend on vendor exceptions."""
