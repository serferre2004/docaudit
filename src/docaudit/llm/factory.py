"""Provider factory.   >>> YOUR TURN: implement ``create_provider``. <<<

Tests that define "done": tests/llm/test_providers.py  (the ``factory`` ones)
"""

from docaudit.config import Settings
from docaudit.llm.base import LLMProvider


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
    raise NotImplementedError
