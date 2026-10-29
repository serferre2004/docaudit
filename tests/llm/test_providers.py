import json

import httpx
import httpx2
import pytest
from pydantic import SecretStr

from docaudit.config import Settings
from docaudit.llm import FakeProvider, LLMError, Message, create_provider
from docaudit.llm import anthropic_provider as anth
from docaudit.llm.ollama_provider import OllamaProvider
from docaudit.llm.ollama_provider import build_payload as ollama_payload

MSGS = [Message(role="user", content="hi")]


# --- Anthropic payload translation (pure function, no network) -----------------


def test_anthropic_puts_system_at_top_level_and_omits_unset_temperature() -> None:
    p = anth.build_payload(MSGS, model="m", system="be terse", temperature=None, max_tokens=50)
    assert p["system"] == "be terse"
    assert p["max_tokens"] == 50
    assert "temperature" not in p
    assert p["messages"] == [{"role": "user", "content": "hi"}]


def test_anthropic_sends_temperature_when_given() -> None:
    p = anth.build_payload(MSGS, model="m", system=None, temperature=0.0, max_tokens=10)
    assert p["temperature"] == 0.0
    assert "system" not in p


# --- Anthropic end-to-end against a mocked HTTP transport  ---------


async def test_anthropic_parses_response() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        assert body["model"] == "anthropic-test"
        return httpx2.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "anthropic-test",
                "content": [
                    {"type": "text", "text": "This"},
                    {"type": "text", "text": " is"},
                    {"type": "tool_use", "id": "toolu_test", "name": "lookup", "input": {}},
                    {"type": "text", "text": " a test"},
                ],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                },
            },
        )

    provider = anth.AnthropicProvider(
        api_key="fake-api-key",
        model="anthropic-test",
        transport=httpx2.MockTransport(handler),
        max_retries=0,
    )
    reply = await provider.complete(MSGS)
    assert reply.text == "This is a test"
    assert reply.stop_reason == "end_turn"
    assert (reply.usage.input_tokens, reply.usage.output_tokens) == (10, 5)
    await provider.aclose()


async def test_anthropic_wraps_http_errors_in_llmerror() -> None:
    transport = httpx2.MockTransport(lambda request: httpx2.Response(500, text="boom"))
    provider = anth.AnthropicProvider(
        api_key="fake-api-key", model="m", transport=transport, max_retries=0
    )
    with pytest.raises(LLMError):
        await provider.complete(MSGS)
    await provider.aclose()


# --- Ollama payload translation -------------------------------------------------


def test_ollama_puts_system_in_messages_and_params_in_options() -> None:
    p = ollama_payload(MSGS, model="m", system="be terse", temperature=0.2, max_tokens=50)
    assert p["messages"][0] == {"role": "system", "content": "be terse"}
    assert p["messages"][1] == {"role": "user", "content": "hi"}
    assert p["options"] == {"num_predict": 50, "temperature": 0.2}
    assert p["stream"] is False


# --- Ollama end-to-end against a mocked HTTP transport ---------------------------


async def test_ollama_parses_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path == "/api/chat"
        assert body["model"] == "llama-test"
        return httpx.Response(
            200,
            json={
                "model": "llama-test",
                "message": {"role": "assistant", "content": "hello!"},
                "done_reason": "stop",
                "prompt_eval_count": 7,
                "eval_count": 3,
            },
        )

    provider = OllamaProvider(
        base_url="http://test", model="llama-test", transport=httpx.MockTransport(handler)
    )
    reply = await provider.complete(MSGS)
    assert reply.text == "hello!"
    assert reply.stop_reason == "stop"
    assert (reply.usage.input_tokens, reply.usage.output_tokens) == (7, 3)
    await provider.aclose()


async def test_ollama_wraps_http_errors_in_llmerror() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(500, text="boom"))
    provider = OllamaProvider(base_url="http://test", model="m", transport=transport)
    with pytest.raises(LLMError):
        await provider.complete(MSGS)
    await provider.aclose()


# --- FakeProvider ---------------------------------------------------------------


async def test_fake_provider_replays_and_records() -> None:
    fake = FakeProvider(replies=["one", "two"])
    assert (await fake.complete(MSGS, system="s")).text == "one"
    assert (await fake.complete(MSGS)).text == "two"
    assert fake.calls[0].system == "s"
    with pytest.raises(LLMError):
        await fake.complete(MSGS)


# --- Factory --------------------------------------------------------------------


def test_factory_requires_api_key_for_anthropic() -> None:
    settings = Settings(llm_provider="anthropic", anthropic_api_key=None, _env_file=None)  # type: ignore[call-arg]
    with pytest.raises(LLMError):
        create_provider(settings)


async def test_factory_builds_selected_provider() -> None:
    async with (
        create_provider(
            Settings(llm_provider="anthropic", anthropic_api_key=SecretStr("k"), _env_file=None)  # type: ignore[call-arg]
        ) as a,
        create_provider(Settings(llm_provider="ollama", _env_file=None)) as o,  # type: ignore[call-arg]
    ):
        assert (a.name, o.name) == ("anthropic", "ollama")


# --- Lifecycle: providers close themselves when the `async with` block exits -----


async def test_ollama_closes_on_exit_even_if_the_block_raises() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={}))
    provider = OllamaProvider(base_url="http://test", model="m", transport=transport)
    with pytest.raises(RuntimeError):
        async with provider as p:
            assert p is provider
            raise RuntimeError("boom")
    assert provider.client.is_closed


async def test_anthropic_closes_on_exit() -> None:
    transport = httpx2.MockTransport(lambda request: httpx2.Response(200, json={}))
    async with anth.AnthropicProvider(api_key="k", model="m", transport=transport) as provider:
        assert not provider.client.is_closed()
    assert provider.client.is_closed()


async def test_fake_provider_rejects_use_after_close() -> None:
    async with FakeProvider(replies=["one", "two"]) as fake:
        assert (await fake.complete(MSGS)).text == "one"
    assert fake.closed
    with pytest.raises(LLMError):
        await fake.complete(MSGS)


# --- Integration (real model; skipped by default) --------------------------------


@pytest.mark.integration
async def test_real_provider_roundtrip() -> None:
    from docaudit.config import get_settings

    async with create_provider(get_settings()) as provider:
        reply = await provider.complete(
            [Message(role="user", content="Reply with exactly the word: pong")], max_tokens=20
        )
    assert "pong" in reply.text.lower()
