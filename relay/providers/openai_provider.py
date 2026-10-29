"""OpenAI chat completions with function calling."""

from __future__ import annotations

import json
from typing import Any

from relay.providers.base import (
    MissingDependencyError,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ProviderError,
    ToolCall,
)


def to_openai_messages(request: ModelRequest) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": request.system}]
    for msg in request.messages:
        if msg.role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": msg.content or None}
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": json.dumps(call.args)},
                    }
                    for call in msg.tool_calls
                ]
            messages.append(entry)
        elif msg.role == "tool":
            messages.append(
                {"role": "tool", "tool_call_id": msg.tool_call_id, "content": msg.content}
            )
        else:
            messages.append({"role": msg.role, "content": msg.content})
    return messages


class OpenAIProvider(ModelProvider):
    name = "openai"

    def __init__(self, model: str, timeout_s: float = 60.0, client: Any | None = None) -> None:
        super().__init__(model)
        if client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise MissingDependencyError("openai", "openai") from exc
            client = AsyncOpenAI(timeout=timeout_s)
        self._client = client

    async def complete(self, request: ModelRequest) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": to_openai_messages(request),
            "max_completion_tokens": request.max_tokens,
        }
        if request.tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in request.tools
            ]
        response = await self._client.chat.completions.create(**kwargs)
        if not response.choices:
            raise ProviderError("OpenAI returned no choices")
        message = response.choices[0].message
        calls = []
        for call in message.tool_calls or []:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                # Keep the raw string so schema validation reports it back to the model.
                args = {"__raw__": call.function.arguments}
            calls.append(ToolCall(id=call.id, name=call.function.name, args=args))
        usage = {}
        if response.usage is not None:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }
        return ModelResponse(text=message.content or "", tool_calls=calls, usage=usage)
