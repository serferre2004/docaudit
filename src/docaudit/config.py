"""Application settings, loaded from environment variables and an optional .env file.

All variables are prefixed with ``DOCAUDIT_`` so they never collide with other tools.
"""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DOCAUDIT_",
        env_file=".env",
        extra="ignore",
    )

    llm_provider: Literal["anthropic", "ollama"] = "ollama"

    # SecretStr keeps the key out of logs and reprs by default.
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-haiku-5-5"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    request_timeout_s: float = 120.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
