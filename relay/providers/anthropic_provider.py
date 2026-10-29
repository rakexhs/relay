"""Anthropic Messages API with tool use."""

from __future__ import annotations

from typing import Any

from relay.providers.base import (
    MissingDependencyError,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ProviderError,
    ToolCall,
)


def to_anthropic_messages(request: ModelRequest) -> list[dict[str, Any]]:
    """Convert to Anthropic's shape: tool results ride in user turns as tool_result blocks.

    Consecutive tool results are merged into a single user message, which is what the API
    expects after an assistant turn with several tool_use blocks.
    """
    out: list[dict[str, Any]] = []
    for msg in request.messages:
        if msg.role == "system":
            # Mid-run system notes are sent as user text; the real system prompt is top-level.
            out.append({"role": "user", "content": [{"type": "text", "text": msg.content}]})
        elif msg.role == "assistant":
            blocks: list[dict[str, Any]] = []
            if msg.content:
                blocks.append({"type": "text", "text": msg.content})
            for call in msg.tool_calls:
                blocks.append(
                    {"type": "tool_use", "id": call.id, "name": call.name, "input": call.args}
                )
            out.append({"role": "assistant", "content": blocks})
        elif msg.role == "tool":
            block = {"type": "tool_result", "tool_use_id": msg.tool_call_id, "content": msg.content}
            previous = out[-1] if out else None
            if (
                previous
                and previous["role"] == "user"
                and all(b.get("type") == "tool_result" for b in previous["content"])
            ):
                previous["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        else:
            out.append({"role": "user", "content": [{"type": "text", "text": msg.content}]})
    return out


class AnthropicProvider(ModelProvider):
    name = "anthropic"

    def __init__(self, model: str, timeout_s: float = 60.0, client: Any | None = None) -> None:
        super().__init__(model)
        if client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as exc:
                raise MissingDependencyError("anthropic", "anthropic") from exc
            client = AsyncAnthropic(timeout=timeout_s)
        self._client = client

    async def complete(self, request: ModelRequest) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "system": request.system,
            "messages": to_anthropic_messages(request),
        }
        if request.tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in request.tools
            ]
        response = await self._client.messages.create(**kwargs)
        if response.stop_reason == "refusal":
            raise ProviderError("Anthropic declined the request (stop_reason=refusal)")
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, args=dict(block.input or {})))
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        return ModelResponse(text="\n".join(text_parts), tool_calls=calls, usage=usage)
