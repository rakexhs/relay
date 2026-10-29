"""A ToolClient that reaches the tools through an MCP ClientSession.

The agent graph does not care whether tools are in-process or behind MCP; this adapter maps
MCP results back to the same dict payloads and passes the run id, trajectory and approval id
in the request's `_meta.relay` field.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from typing import Any

from mcp import ClientSession
from mcp.shared.memory import create_connected_server_and_client_session

from relay.approvals import ApprovalStore
from relay.providers.base import ToolSpec
from relay.tools.errors import ToolErrorCode, error_payload
from relay.tools.registry import CallMeta, ToolRegistry


class MCPToolClient:
    def __init__(self, session: ClientSession, approval_tools: Iterable[str]) -> None:
        self.session = session
        self.approval_tools = frozenset(approval_tools)

    async def list_tools(self) -> list[ToolSpec]:
        listed = await self.session.list_tools()
        return [
            ToolSpec(name=t.name, description=t.description or "", parameters=t.inputSchema)
            for t in listed.tools
        ]

    def requires_approval(self, name: str, args: dict[str, Any]) -> bool:
        if name not in self.approval_tools:
            return False
        if name == "draft_reply":
            return args.get("send") is True
        return True

    async def call(self, name: str, args: dict[str, Any], meta: CallMeta) -> dict[str, Any]:
        relay_meta: dict[str, Any] = {"run_id": meta.run_id, "trajectory": meta.trajectory}
        if meta.approval_id:
            relay_meta["approval_id"] = meta.approval_id
        result = await self.session.call_tool(name, args, meta={"relay": relay_meta})
        text = "\n".join(getattr(block, "text", "") for block in result.content)
        if result.isError:
            # FastMCP rejected the call before it reached the registry (e.g. a wrong type).
            return error_payload(ToolErrorCode.INVALID_ARGUMENTS, text or "tool call failed")
        if isinstance(result.structuredContent, dict):
            payload = result.structuredContent
            # FastMCP wraps non-object return annotations as {"result": ...}
            return payload.get("result", payload) if set(payload) == {"result"} else payload
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return error_payload(ToolErrorCode.INTERNAL, f"unparseable tool result: {text[:200]}")


@asynccontextmanager
async def in_process_client(
    registry: ToolRegistry, approvals: ApprovalStore | None, approval_tools: Iterable[str]
) -> AsyncIterator[MCPToolClient]:
    """Run the MCP server and client in this process, connected by memory streams."""
    from relay.mcp_server import build_mcp_server

    server = build_mcp_server(registry, approvals)
    async with create_connected_server_and_client_session(server) as session:
        yield MCPToolClient(session, approval_tools)
