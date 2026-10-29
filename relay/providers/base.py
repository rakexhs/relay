"""Provider-neutral message types and the interface every model provider implements."""

from __future__ import annotations

import abc
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]
Purpose = Literal["plan", "act", "judge", "summarize"]


class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None


class ToolSpec(BaseModel):
    """What the model sees about a tool: name, description and a JSON schema."""

    name: str
    description: str
    parameters: dict[str, Any]


class ModelRequest(BaseModel):
    purpose: Purpose
    system: str
    messages: list[Message]
    tools: list[ToolSpec] = Field(default_factory=list)
    max_tokens: int = 4096


class ModelResponse(BaseModel):
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)


class ProviderError(RuntimeError):
    """Raised when a provider cannot produce a usable response."""


class MissingDependencyError(ProviderError):
    def __init__(self, package: str, extra: str) -> None:
        super().__init__(
            f"The '{package}' package is not installed. Install it with: pip install 'relay[{extra}]'"
        )
        self.package = package
        self.extra = extra


class ModelProvider(abc.ABC):
    """A chat model that can call tools. Implementations must be safe to share across runs."""

    name: str = "base"

    def __init__(self, model: str) -> None:
        self.model = model

    @abc.abstractmethod
    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Return either text, tool calls, or both for the given request."""

    def describe(self) -> dict[str, str]:
        return {"provider": self.name, "model": self.model}
