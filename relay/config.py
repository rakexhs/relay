"""Runtime configuration, read from RELAY_* environment variables (and .env)."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

ProviderName = Literal["stub", "openai", "anthropic"]

WRITE_TOOLS = ("draft_reply", "escalate_to_human")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RELAY_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Model provider
    provider: ProviderName = "stub"
    model: str = ""
    request_timeout_s: float = 60.0

    # Storage
    database_url: str = ""
    checkpointer: Literal["memory", "postgres"] = "memory"
    opensearch_url: str = ""
    log_index: str = "app-logs"

    # Agent loop
    max_steps: int = Field(default=12, ge=1, le=100)
    max_consecutive_errors: int = Field(default=3, ge=1)
    approval_tools: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: list(WRITE_TOOLS)
    )
    quote_retrieved: bool = True
    memory_top_k: int = 3
    embedding_dim: int = 256

    # Voice
    whisper_model: str = "base"

    # Tracing
    otel_exporter: Literal["none", "console", "otlp"] = "none"
    service_name: str = "relay"

    @field_validator("approval_tools", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @property
    def default_model(self) -> str:
        if self.model:
            return self.model
        return {
            "stub": "stub-policy-v1",
            "openai": "gpt-4.1",
            "anthropic": "claude-opus-5-5",
        }[self.provider]

    @property
    def uses_postgres(self) -> bool:
        return bool(self.database_url)


@lru_cache
def get_settings() -> Settings:
    return Settings()
