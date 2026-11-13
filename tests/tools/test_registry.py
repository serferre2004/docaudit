import json

import pytest

from docaudit.tools import Tool, ToolError, ToolRegistry

from .helpers import EchoInput, call, echo_registry, echo_tool


def test_duplicate_names_are_rejected() -> None:
    with pytest.raises(ValueError):
        ToolRegistry([echo_tool(), echo_tool()])


def test_definitions_expose_the_schema_the_model_reads() -> None:
    definition = echo_registry().definitions[0]
    assert (definition.name, definition.description) == ("echo", "Repeats the text.")
    assert definition.input_schema == EchoInput.model_json_schema()


def test_success_returns_json_with_the_call_id() -> None:
    result = echo_registry().execute(call(call_id="abc", text="hi"))
    assert (result.tool_call_id, result.name, result.is_error) == ("abc", "echo", False)
    assert json.loads(result.content) == {"echo": "hi"}


def test_decimal_results_do_not_crash() -> None:
    from decimal import Decimal

    tool = Tool("money", "x", EchoInput, lambda _: {"amount": Decimal("1.50")})
    result = ToolRegistry([tool]).execute(call("money", text="x"))
    assert json.loads(result.content) == {"amount": "1.50"}


def test_unknown_tool_is_an_error_the_model_can_read() -> None:
    result = echo_registry().execute(call("invented", call_id="x"))
    assert result.is_error
    assert result.tool_call_id == "x"
    assert result.name == "invented"
    assert "echo" in result.content, "the model needs to know what it can call instead"


def test_invalid_arguments_never_reach_the_handler() -> None:
    calls: list[str] = []
    result = echo_registry(calls).execute(call(wrong_name="hi"))
    assert result.is_error
    assert "text" in result.content, "name the field that failed"
    assert calls == []


def test_tool_error_becomes_a_result() -> None:
    def handler(_: EchoInput) -> dict[str, str]:
        raise ToolError("no rate for XYZ")

    result = ToolRegistry([Tool("t", "x", EchoInput, handler)]).execute(call("t", text="a"))
    assert result.is_error
    assert "no rate for XYZ" in result.content


def test_a_bug_in_a_handler_is_not_the_models_mistake() -> None:
    def handler(_: EchoInput) -> dict[str, str]:
        raise KeyError("oops")

    with pytest.raises(KeyError):
        ToolRegistry([Tool("t", "x", EchoInput, handler)]).execute(call("t", text="a"))
