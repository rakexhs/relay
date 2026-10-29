"""Model providers. Pick one with RELAY_PROVIDER=stub|openai|anthropic."""

from __future__ import annotations

from relay.config import Settings
from relay.providers.base import (
    Message,
    MissingDependencyError,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ProviderError,
    ToolCall,
    ToolSpec,
)
from relay.providers.stub import StubProvider

__all__ = [
    "Message",
    "MissingDependencyError",
    "ModelProvider",
    "ModelRequest",
    "ModelResponse",
    "ProviderError",
    "StubProvider",
    "ToolCall",
    "ToolSpec",
    "get_provider",
]


def get_provider(settings: Settings, name: str | None = None) -> ModelProvider:
    name = name or settings.provider
    model = settings.model if name == settings.provider else ""
    if name == "stub":
        return StubProvider(model or "stub-policy-v1")
    if name == "openai":
        from relay.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(model or "gpt-4.1", timeout_s=settings.request_timeout_s)
    if name == "anthropic":
        from relay.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(model or "claude-opus-5-5", timeout_s=settings.request_timeout_s)
    raise ValueError(f"Unknown provider {name!r}; expected stub, openai or anthropic")
