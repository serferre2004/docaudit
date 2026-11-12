import pytest

from docaudit.llm import FakeProvider, LLMError, LLMResponse, Message, Usage
from docaudit.tools import ToolLoopError, run_tool_loop

from .helpers import asks, call, echo_registry

QUESTION = [Message(role="user", content="go")]


async def test_a_reply_without_calls_ends_at_once() -> None:
    fake = FakeProvider(replies=["done"])
    result = await run_tool_loop(fake, echo_registry(), QUESTION, system="be good", max_tokens=99)
    assert result.text == "done"
    assert (result.turns, result.executions) == (1, [])
    sent = fake.calls[0]
    assert (sent.system, sent.temperature, sent.max_tokens) == ("be good", 0.0, 99)
    assert sent.tools == echo_registry().definitions


async def test_runs_the_tool_and_sends_the_result_back() -> None:
    ran: list[str] = []
    fake = FakeProvider(replies=[asks(call(text="hi"), text="checking"), "the end"])
    result = await run_tool_loop(fake, echo_registry(ran), QUESTION)

    assert ran == ["hi"]
    assert (result.text, result.turns) == ("the end", 2)
    history = fake.calls[1].messages
    assert [m.role for m in history] == ["user", "assistant", "user"]
    assert history[1].content == "checking"
    assert history[1].tool_calls[0].id == "c1"
    assert history[2].tool_results[0].tool_call_id == "c1"
    assert '"hi"' in history[2].tool_results[0].content


async def test_decides_by_tool_calls_not_stop_reason() -> None:
    ollama_style = LLMResponse(
        text="", model="m", stop_reason="stop", tool_calls=(call(text="hi"),)
    )
    ran: list[str] = []
    fake = FakeProvider(replies=[ollama_style, "the end"])
    assert (await run_tool_loop(fake, echo_registry(ran), QUESTION)).text == "the end"
    assert ran == ["hi"]


async def test_every_parallel_call_gets_its_answer_in_one_message() -> None:
    ran: list[str] = []
    both = asks(call(call_id="a", text="one"), call(call_id="b", text="two"))
    fake = FakeProvider(replies=[both, "the end"])
    result = await run_tool_loop(fake, echo_registry(ran), QUESTION)

    assert ran == ["one", "two"]
    last = fake.calls[1].messages[-1]
    assert [r.tool_call_id for r in last.tool_results] == ["a", "b"]
    assert [e.call.id for e in result.executions] == ["a", "b"]


async def test_tool_errors_go_back_to_the_model_instead_of_stopping() -> None:
    fake = FakeProvider(replies=[asks(call("invented")), asks(call(text="ok")), "recovered"])
    result = await run_tool_loop(fake, echo_registry(), QUESTION)
    assert result.text == "recovered"
    assert result.executions[0].result.is_error
    assert not result.executions[1].result.is_error


async def test_gives_up_after_max_turns() -> None:
    fake = FakeProvider(replies=[asks(call(text="again"))] * 10)
    with pytest.raises(ToolLoopError, match="3"):
        await run_tool_loop(fake, echo_registry(), QUESTION, max_turns=3)
    assert len(fake.calls) == 3


@pytest.mark.parametrize("reason", ["max_tokens", "length"])
async def test_a_truncated_reply_is_not_trusted(reason: str) -> None:
    ran: list[str] = []
    cut = LLMResponse(text="", model="m", stop_reason=reason, tool_calls=(call(text="hi"),))
    with pytest.raises(ToolLoopError):
        await run_tool_loop(FakeProvider(replies=[cut]), echo_registry(ran), QUESTION)
    assert ran == [], "a tool must not run with arguments that may be cut off"


async def test_truncated_final_text_is_not_an_answer() -> None:
    cut = LLMResponse(text="The total is", model="m", stop_reason="length")
    with pytest.raises(ToolLoopError):
        await run_tool_loop(FakeProvider(replies=[cut]), echo_registry(), QUESTION)


async def test_usage_is_summed_over_all_turns() -> None:
    fake = FakeProvider(
        replies=[asks(call(text="a")), asks(call(text="b")), "end"],
        usage_per_call=Usage(input_tokens=10, output_tokens=2),
    )
    result = await run_tool_loop(fake, echo_registry(), QUESTION)
    assert result.usage == Usage(input_tokens=30, output_tokens=6)


async def test_the_callers_messages_are_not_changed() -> None:
    mine = list(QUESTION)
    fake = FakeProvider(replies=[asks(call(text="a")), "end"])
    result = await run_tool_loop(fake, echo_registry(), mine)
    assert mine == QUESTION
    assert len(result.messages) > len(mine)


async def test_provider_errors_propagate() -> None:
    with pytest.raises(LLMError):
        await run_tool_loop(FakeProvider(replies=[]), echo_registry(), QUESTION)


async def test_rejects_max_turns_below_one() -> None:
    with pytest.raises(ValueError):
        await run_tool_loop(FakeProvider(replies=[]), echo_registry(), QUESTION, max_turns=0)
