"""Ollama (local models) backend.   >>> YOUR TURN: implement this file. <<<

Docs: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion
Tests that define "done": tests/llm/test_providers.py  (the ``ollama`` ones)

We deliberately use ``httpx`` directly instead of a client library: an LLM API is just a JSON
POST, and ``httpx.MockTransport`` lets us test the provider with no server at all.
"""

from collections.abc import Sequence
from typing import Any

import httpx

from docaudit.llm.types import LLMError, LLMResponse, Message, Usage


def build_payload(
    messages: Sequence[Message],
    *,
    model: str,
    system: str | None,
    temperature: float | None,
    max_tokens: int,
) -> dict[str, Any]:
    """Translate our neutral types into the JSON body for ``POST /api/chat``.

    Requirements (note how they differ from Anthropic!):
      * The system prompt is NOT a separate field: it becomes the FIRST message with
        ``role: "system"`` (only if one was given).
      * ``stream`` must be ``False`` so we get a single JSON response.
      * Sampling params go under ``options``: ``num_predict`` is Ollama's name for max tokens.
        Include ``temperature`` in ``options`` ONLY if it is not ``None``.
    """
    msg_list = []
    if system:
        msg_list.append({"role": "system", "content": system})
    for msg in messages:
        msg_list.append(msg.model_dump())
    options: dict[str, object] = {"num_predict": max_tokens}
    if temperature is not None:
        options["temperature"] = temperature
    base_dict = {
        "model": model,
        "stream": False,
        "options": options,
        "messages": msg_list,
    }
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
        )

    async def aclose(self) -> None:
        await self.client.aclose()
