"""Provider factory.   >>> YOUR TURN: implement ``create_provider``. <<<

Tests that define "done": tests/llm/test_providers.py  (the ``factory`` ones)
"""

from docaudit.config import Settings
from docaudit.llm.anthropic_provider import AnthropicProvider
from docaudit.llm.base import LLMProvider
from docaudit.llm.ollama_provider import OllamaProvider
from docaudit.llm.types import LLMError


def create_provider(settings: Settings) -> LLMProvider:
    """Build the provider selected by ``settings.llm_provider``.

    Requirements:
      * "anthropic" -> ``AnthropicProvider``. If ``settings.anthropic_api_key`` is ``None``,
        raise ``LLMError`` with a message that tells the user which env var to set.
        (``anthropic_api_key`` is a ``SecretStr``: look up how to read its value safely.)
      * "ollama" -> ``OllamaProvider``.
      * Pass ``model`` and ``timeout_s`` through from settings.

    Design note: this is the ONLY function in the codebase allowed to know the concrete
    provider classes. Everything else depends on the ``LLMProvider`` protocol.
    """
    match settings.llm_provider:
        case "anthropic":
            if settings.anthropic_api_key is None:
                raise LLMError(
                    "Anthropic API key missing: set DOCAUDIT_ANTHROPIC_API_KEY in your environment "
                    "or .env file"
                )
            return AnthropicProvider(
                api_key=settings.anthropic_api_key.get_secret_value(),
                model=settings.anthropic_model,
                timeout_s=settings.request_timeout_s,
            )
        case "ollama":
            return OllamaProvider(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                timeout_s=settings.request_timeout_s,
            )
        case _:
            raise LLMError(
                f"Unknown LLM provider {settings.llm_provider!r}: set DOCAUDIT_LLM_PROVIDER "
                "to 'anthropic' or 'ollama'"
            )
