import json

import httpx
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


def test_factory_builds_selected_provider() -> None:
    a = create_provider(
        Settings(llm_provider="anthropic", anthropic_api_key=SecretStr("k"), _env_file=None)  # type: ignore[call-arg]
    )
    o = create_provider(Settings(llm_provider="ollama", _env_file=None))  # type: ignore[call-arg]
    assert (a.name, o.name) == ("anthropic", "ollama")


# --- Integration (real model; skipped by default) --------------------------------


@pytest.mark.integration
async def test_real_provider_roundtrip() -> None:
    from docaudit.config import get_settings

    provider = create_provider(get_settings())
    reply = await provider.complete(
        [Message(role="user", content="Reply with exactly the word: pong")], max_tokens=20
    )
    assert "pong" in reply.text.lower()
