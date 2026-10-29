"""The thin layer the agent uses to reach its tools.

The graph only talks to a `ToolClient`. `LocalToolClient` calls the registry in-process (tests,
evals, the API's default). `relay.mcp_client.MCPToolClient` reaches the same tools through an
MCP session. Both return the same result and error payloads.
"""

from __future__ import annotations

from typing import Any, Protocol

from relay.providers.base import ToolSpec
from relay.tools.registry import CallMeta, ToolRegistry


class ToolClient(Protocol):
    async def list_tools(self) -> list[ToolSpec]: ...

    def requires_approval(self, name: str, args: dict[str, Any]) -> bool: ...

    async def call(self, name: str, args: dict[str, Any], meta: CallMeta) -> dict[str, Any]: ...


class LocalToolClient:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    async def list_tools(self) -> list[ToolSpec]:
        return self.registry.specs()

    def requires_approval(self, name: str, args: dict[str, Any]) -> bool:
        parsed = self.registry.validate(name, args)
        if isinstance(parsed, dict):
            return False  # invalid calls never run, so there is nothing to approve
        return self.registry.requires_approval(name, parsed)

    async def call(self, name: str, args: dict[str, Any], meta: CallMeta) -> dict[str, Any]:
        return await self.registry.invoke(name, args, meta)
