"""Getting a JSON object out of LLM text.   >>> YOUR TURN: implement this file. <<<

Tests that define "done": tests/extraction/test_parsing.py

Even when told "reply with JSON only", models wrap it in Markdown fences or add a sentence
before or after. This is the tolerant step between "text the model wrote" and "data we validate".
"""

from typing import Any


def parse_json_object(text: str) -> dict[str, Any]:
    """Return the first JSON *object* found in ``text``.

    Requirements:
      * Accept a bare object: ``'{"a": 1}'``.
      * Accept an object inside a Markdown fence (with or without the ``json`` tag).
      * Accept an object surrounded by prose: ``'Sure! {"a": 1} Hope this helps.'``.
      * Only the FIRST JSON value counts: ``'{"a": 1} and {"b": 2}'`` gives ``{"a": 1}``, and
        ``'[{"a": 1}]'`` is an error because that first value is a list, not an object.
      * Raise ``ExtractionError`` (from ``docaudit.extraction.schemas``) when there is no JSON
        value, when it is malformed, or when it is not an object.
        Never let a ``json.JSONDecodeError`` escape: callers only know about ``ExtractionError``.
      * The error message must say what went wrong: it is fed back to the model to repair itself.

    Keep this a pure function. Hint: look at ``json.JSONDecoder.raw_decode`` in the stdlib docs.
    """
    raise NotImplementedError
