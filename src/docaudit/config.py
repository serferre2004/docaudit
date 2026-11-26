"""Application settings, loaded from environment variables and an optional .env file.

All variables are prefixed with ``DOCAUDIT_`` so they never collide with other tools.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DOCAUDIT_",
        env_file=".env",
        extra="ignore",
        # DOCAUDIT_API_KEY= (set but empty) means unset, not "the key is the empty string".
        env_ignore_empty=True,
    )

    llm_provider: Literal["anthropic", "ollama"] = "ollama"

    # SecretStr keeps the key out of logs and reprs by default.
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-haiku-5-5"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_embedding_model: str = "nomic-embed-text"

    request_timeout_s: float = 120.0

    # --- API (Lesson 6) ---
    # Clients send it in the X-API-Key header. None disables auth: fine on your laptop, never
    # on a server.
    api_key: SecretStr | None = None
    # Where the API keeps its checkpoints. Docker mounts a volume here so they survive restarts.
    api_db_path: Path = Path("data/generated/api-checkpoints.sqlite")


@lru_cache
def get_settings() -> Settings:
    return Settings()
