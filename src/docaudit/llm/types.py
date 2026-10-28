"""Provider-neutral data types.

These are the only LLM-related types the rest of the codebase should know about.
Each provider translates to and from its own wire format.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

Role = Literal["user", "assistant"]


class Message(BaseModel):
    """One turn of a conversation.

    The system prompt is *not* a message: providers disagree on where it goes
    (Anthropic has a top-level field, Ollama uses a ``system`` role), so it is
    passed separately and each provider places it correctly.
    """

    model_config = ConfigDict(frozen=True)

    role: Role
    content: str


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


class LLMError(Exception):
    """Raised for any provider failure, so callers never depend on vendor exceptions."""
