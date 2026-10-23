"""Configuration.

Every setting is an environment variable with a MEM_ prefix, loaded from the
project-root .env file (gitignored). Scope rule of the project: user_id and
thread_id always come from the application, never from the model — they are
function arguments here, not prompt content.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = src/agentmem/settings.py -> parents[2] == agent-memory/
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MEM_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenRouter (OpenAI-compatible; one key covers chat and embeddings)
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    api_key: str | None = None

    # Note: OpenRouter model IDs are provider-prefixed (openai/gpt-..., ...).
    chat_model: str = "openai/gpt-4o-mini"
    embed_model: str = "openai/text-embedding-3-small"
    embed_dims: int = 1536

    # Cross-family judge for the evaluation harness (may stay unset in early
    # milestones; the eval falls back to the chat model and reports that).
    judge_model: str | None = None

    # 127.0.0.1, not localhost: Docker maps the port on IPv4 only here, and
    # libpq's first attempt on ::1 can hang instead of failing fast.
    database_url: str = "postgresql://acme:acme_local_dev@127.0.0.1:15432/agentmem"
    database_url_lg: str = "postgresql://acme:acme_local_dev@127.0.0.1:15432/agentmem_lg"

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)


def get_settings() -> Settings:
    return Settings()
