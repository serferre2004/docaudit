"""The tool-use loop, by hand.   >>> YOUR TURN: implement this file. <<<

Tests that define "done": tests/tools/test_loop.py

The whole trick fits in a few lines: the model asks, your code runs it, the result goes back in the
history, and you call the model again. The model stays as stateless as it was in Lesson 1.
"""

from collections.abc import Sequence

from docaudit.llm.base import LLMProvider
from docaudit.llm.types import Message, Usage
from docaudit.tools.registry import ToolRegistry
from docaudit.tools.schemas import ToolExecution, ToolLoopError, ToolLoopResult

# Providers name this differently: "max_tokens" on Anthropic, "length" on Ollama.
TRUNCATED_STOP_REASONS = frozenset({"max_tokens", "length"})


async def run_tool_loop(
    provider: LLMProvider,
    registry: ToolRegistry,
    messages: Sequence[Message],
    *,
    system: str | None = None,
    max_turns: int = 10,
    temperature: float | None = 0.0,
    max_tokens: int = 1024,
) -> ToolLoopResult:
    """Call the model, running the tools it asks for, until it answers without asking for any.

    Requirements:
      * ``max_turns < 1`` raises ``ValueError``. A turn is one call to the model.
      * Work on a copy of ``messages``: the caller's list must not change.
      * Each turn calls ``provider.complete`` with ``system``, ``temperature``, ``max_tokens`` and
        ``tools=registry.definitions``, and adds up ``usage``.
      * A reply with ``stop_reason`` in ``TRUNCATED_STOP_REASONS`` raises ``ToolLoopError``: its
        text or tool arguments may be cut off, so neither can be trusted.
      * A reply with no ``tool_calls`` ends the loop and returns a ``ToolLoopResult``. Decide this
        by looking at ``tool_calls``, NOT ``stop_reason``: Ollama reports ``"stop"`` either way.
      * Otherwise:
          1. Append the reply as an ``assistant`` message, with its text and ``tool_calls``.
          2. Run EVERY call with ``registry.execute``, in order, and keep a ``ToolExecution`` for
             each. A model can ask for several at once, and each one must get its answer.
          3. Append ONE ``user`` message with all the ``tool_results``.
      * Turns run out -> raise ``ToolLoopError`` saying how many turns were made. A model that
        keeps asking must not run (and bill you) forever.
      * ``LLMError`` is not caught.
    """
    if max_turns < 1:
        raise ValueError(f"max_turns must be at least 1, {max_turns} got")
    messages = list(messages)
    input_tokens = 0
    output_tokens = 0
    executions: list[ToolExecution] = []
    for turn in range(1, max_turns + 1):
        response = await provider.complete(
            messages,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=registry.definitions,
        )
        input_tokens += response.usage.input_tokens
        output_tokens += response.usage.output_tokens
        if response.stop_reason in TRUNCATED_STOP_REASONS:
            raise ToolLoopError("Response from LLM is truncated.")
        calls = response.tool_calls
        if not calls:
            return ToolLoopResult(
                text=response.text,
                messages=messages,
                executions=executions,
                turns=turn,
                usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
            )
        reply_message = Message(role="assistant", content=response.text, tool_calls=calls)
        messages.append(reply_message)
        tool_results = []
        for call in calls:
            result = registry.execute(call)
            tool_results.append(result)
            execution = ToolExecution(call=call, result=result)
            executions.append(execution)
        result_message = Message(role="user", tool_results=tool_results)
        messages.append(result_message)
    raise ToolLoopError(f"LLM couldn't get enough information using tools, {turn} turns were made.")
