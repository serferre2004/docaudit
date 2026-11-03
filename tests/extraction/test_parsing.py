import pytest

from docaudit.extraction import ExtractionError
from docaudit.extraction.parsing import parse_json_object

OBJ = '{"a": 1, "b": [1, 2], "c": {"d": "}"}}'
EXPECTED = {"a": 1, "b": [1, 2], "c": {"d": "}"}}


def test_parses_a_bare_object() -> None:
    assert parse_json_object(OBJ) == EXPECTED


@pytest.mark.parametrize(
    "text",
    [
        f"```json\n{OBJ}\n```",
        f"```\n{OBJ}\n```",
        f"Sure! Here is the data:\n{OBJ}\nHope this helps.",
        f"  \n{OBJ}\n\n",
    ],
    ids=["json-fence", "plain-fence", "prose-around", "whitespace"],
)
def test_tolerates_fences_and_prose(text: str) -> None:
    assert parse_json_object(text) == EXPECTED


def test_only_the_first_value_counts() -> None:
    assert parse_json_object('{"a": 1} and then {"b": 2}') == {"a": 1}


@pytest.mark.parametrize(
    "text",
    [
        "I could not find an invoice in this document.",
        '{"a": 1,',
        "{'a': 1}",
        '[{"a": 1}]',
        "",
    ],
    ids=["no-json", "truncated", "single-quotes", "top-level-list", "empty"],
)
def test_raises_extraction_error_with_a_message(text: str) -> None:
    with pytest.raises(ExtractionError) as excinfo:
        parse_json_object(text)
    assert str(excinfo.value), "the message is sent back to the model: it must say what failed"
