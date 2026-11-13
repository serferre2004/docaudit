"""The same tool conversation, in the two wire formats. No model involved."""

import json
from typing import Any

import httpx
import httpx2

from docaudit.llm import Message, ToolCall, ToolDefinition, ToolResult
from docaudit.llm import anthropic_provider as anth
from docaudit.llm import ollama_provider as oll

SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}}}
TOOL = ToolDefinition(name="lookup", description="Finds things.", input_schema=SCHEMA)
CALL = ToolCall(id="toolu_1", name="lookup", arguments={"name": "Acme"})
PLAIN = Message(role="user", content="hi")
ASKS = Message(role="assistant", content="Checking.", tool_calls=(CALL,))
ANSWERS = Message(
    role="user",
    tool_results=(
        ToolResult(tool_call_id="toolu_1", name="lookup", content='{"found": true}'),
        ToolResult(tool_call_id="toolu_2", name="lookup", content="no rate", is_error=True),
    ),
)


# --- Anthropic ---------------------------------------------------------------------


def test_anthropic_tool_shape() -> None:
    assert anth.tools_to_wire([TOOL]) == [
        {"name": "lookup", "description": "Finds things.", "input_schema": SCHEMA}
    ]


def test_anthropic_plain_message_is_unchanged() -> None:
    assert anth.message_to_wire(PLAIN) == {"role": "user", "content": "hi"}


def test_anthropic_assistant_turn_is_a_list_of_blocks() -> None:
    assert anth.message_to_wire(ASKS) == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Checking."},
            {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"name": "Acme"}},
        ],
    }


def test_anthropic_omits_empty_text_blocks() -> None:
    wire = anth.message_to_wire(Message(role="assistant", tool_calls=(CALL,)))
    assert [block["type"] for block in wire["content"]] == ["tool_use"]


def test_anthropic_results_are_blocks_in_a_user_turn() -> None:
    wire = anth.message_to_wire(ANSWERS)
    assert wire["role"] == "user"
    assert wire["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_1",
            "content": '{"found": true}',
            "is_error": False,
        },
        {"type": "tool_result", "tool_use_id": "toolu_2", "content": "no rate", "is_error": True},
    ]


def test_anthropic_payload_includes_tools_only_when_given() -> None:
    args: dict[str, Any] = {"model": "m", "system": None, "temperature": None, "max_tokens": 10}
    assert "tools" not in anth.build_payload([PLAIN], **args)
    payload = anth.build_payload([PLAIN, ASKS, ANSWERS], tools=[TOOL], **args)
    assert payload["tools"][0]["name"] == "lookup"
    assert len(payload["messages"]) == 3
    assert payload["messages"][1]["content"][1]["type"] == "tool_use"


async def test_anthropic_end_to_end() -> None:
    sent: list[dict[str, object]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "content": [
                    {"type": "text", "text": "Let me check."},
                    {"type": "tool_use", "id": "toolu_9", "name": "lookup", "input": {"name": "A"}},
                ],
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {"input_tokens": 5, "output_tokens": 3},
            },
        )

    provider = anth.AnthropicProvider(
        api_key="k", model="claude-test", transport=httpx2.MockTransport(handler), max_retries=0
    )
    async with provider:
        reply = await provider.complete([PLAIN], tools=[TOOL])
    assert sent[0]["tools"] == anth.tools_to_wire([TOOL])
    assert reply.text == "Let me check."
    assert reply.stop_reason == "tool_use"
    assert reply.tool_calls == (ToolCall(id="toolu_9", name="lookup", arguments={"name": "A"}),)


async def test_anthropic_reply_without_tools_has_no_calls() -> None:
    def handler(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    provider = anth.AnthropicProvider(
        api_key="k", model="claude-test", transport=httpx2.MockTransport(handler), max_retries=0
    )
    async with provider:
        assert (await provider.complete([PLAIN])).tool_calls == ()


# --- Ollama ------------------------------------------------------------------------


def test_ollama_tool_shape() -> None:
    assert oll.tools_to_wire([TOOL]) == [
        {
            "type": "function",
            "function": {"name": "lookup", "description": "Finds things.", "parameters": SCHEMA},
        }
    ]


def test_ollama_plain_message_is_unchanged() -> None:
    assert oll.message_to_wire(PLAIN) == [{"role": "user", "content": "hi"}]


def test_ollama_assistant_turn_carries_the_calls() -> None:
    assert oll.message_to_wire(ASKS) == [
        {
            "role": "assistant",
            "content": "Checking.",
            "tool_calls": [{"function": {"name": "lookup", "arguments": {"name": "Acme"}}}],
        }
    ]


def test_ollama_each_result_is_its_own_message_and_errors_are_marked() -> None:
    assert oll.message_to_wire(ANSWERS) == [
        {"role": "tool", "tool_name": "lookup", "content": '{"found": true}'},
        {"role": "tool", "tool_name": "lookup", "content": "ERROR: no rate"},
    ]


def test_ollama_parses_calls_and_makes_up_missing_ids() -> None:
    calls = oll.parse_tool_calls(
        {
            "content": "",
            "tool_calls": [
                {"function": {"name": "a", "arguments": {"x": 1}}},
                {"id": "real", "function": {"name": "b", "arguments": {}}},
            ],
        }
    )
    assert [(c.id, c.name, c.arguments) for c in calls] == [
        ("call_0", "a", {"x": 1}),
        ("real", "b", {}),
    ]
    assert oll.parse_tool_calls({"content": "hi"}) == ()


def test_ollama_payload_flattens_results_after_the_system_message() -> None:
    payload = oll.build_payload(
        [PLAIN, ASKS, ANSWERS],
        model="m",
        system="s",
        temperature=None,
        max_tokens=10,
        tools=[TOOL],
    )
    assert [m["role"] for m in payload["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "tool",
    ]
    assert payload["tools"][0]["function"]["name"] == "lookup"
    bare = oll.build_payload([PLAIN], model="m", system=None, temperature=None, max_tokens=10)
    assert "tools" not in bare


async def test_ollama_end_to_end() -> None:
    sent: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "llama",
                "done_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "lookup", "arguments": {"name": "A"}}}],
                },
            },
        )

    provider = oll.OllamaProvider(
        base_url="http://test", model="llama", transport=httpx.MockTransport(handler)
    )
    async with provider:
        reply = await provider.complete([PLAIN], tools=[TOOL])
    assert sent[0]["tools"] == oll.tools_to_wire([TOOL])
    assert reply.stop_reason == "stop", "Ollama doesn't say tool_use: look at tool_calls instead"
    assert reply.tool_calls == (ToolCall(id="call_0", name="lookup", arguments={"name": "A"}),)
