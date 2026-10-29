import asyncio
import json
from types import SimpleNamespace

import pytest

from relay.config import Settings
from relay.providers import Message, ModelRequest, StubProvider, ToolCall, ToolSpec, get_provider
from relay.providers.anthropic_provider import AnthropicProvider, to_anthropic_messages
from relay.providers.openai_provider import OpenAIProvider, to_openai_messages

TICKET = {
    "ticket_id": "T-1",
    "customer_id": "C-1",
    "subject": "Webhooks returning 502 since yesterday",
    "body": "Our webhook endpoint stopped receiving events and we see 502 errors.",
}


def _history(*steps: tuple[ToolCall, dict]) -> list[Message]:
    messages = [Message(role="user", content=f"New ticket:\n{json.dumps(TICKET)}")]
    for call, result in steps:
        messages.append(Message(role="assistant", tool_calls=[call]))
        messages.append(
            Message(role="tool", tool_call_id=call.id, name=call.name, content=json.dumps(result))
        )
    return messages


def _act(messages):
    request = ModelRequest(purpose="act", system="", messages=messages)
    return asyncio.run(StubProvider().complete(request))


def test_stub_starts_with_lookup():
    response = _act(_history())
    assert response.tool_calls[0].name == "lookup_ticket"
    assert response.tool_calls[0].args == {"ticket_id": "T-1"}


def test_stub_repairs_log_query_from_structured_error():
    lookup = ToolCall(name="lookup_ticket", args={"ticket_id": "T-1"})
    docs = ToolCall(name="search_docs", args={"query": "webhooks"})
    first = _act(
        _history(
            (lookup, {"ticket": {}, "customer": {"name": "Ana"}}),
            (
                docs,
                {"results": [{"doc_id": "D1", "title": "Webhooks", "snippet": "x", "score": 0.9}]},
            ),
        )
    ).tool_calls[0]
    assert first.name == "query_logs"
    assert "status:502" in first.args["query"]
    assert first.args["since"] == "yesterday"

    error = {"error": {"code": "invalid_arguments", "field": "since", "hint": "Use 15m, 2h, 7d"}}
    second = _act(
        _history(
            (lookup, {"customer": {"name": "Ana"}}),
            (docs, {"results": [{"doc_id": "D1", "title": "W", "snippet": "x", "score": 0.9}]}),
            (first, error),
        )
    ).tool_calls[0]
    assert second.args["since"] == "48h"

    error = {
        "error": {
            "code": "invalid_query",
            "field": "query",
            "hint": "Unknown field 'status'",
            "allowed": ["service", "level", "status_code", "customer_id", "message"],
        }
    }
    third = _act(
        _history(
            (lookup, {"customer": {"name": "Ana"}}),
            (docs, {"results": [{"doc_id": "D1", "title": "W", "snippet": "x", "score": 0.9}]}),
            (first, {"error": {"code": "invalid_arguments", "field": "since", "hint": "h"}}),
            (second, error),
        )
    ).tool_calls[0]
    assert "status_code:502" in third.args["query"]


def test_stub_escalates_when_a_tool_keeps_failing():
    lookup = ToolCall(name="lookup_ticket", args={"ticket_id": "T-1"})
    down = {"error": {"code": "backend_unavailable", "field": None, "hint": "retry later"}}
    response = _act(_history((lookup, down), (lookup, down)))
    call = response.tool_calls[0]
    assert call.name == "escalate_to_human"
    assert call.args["reason"] == "tool_failure"


def test_stub_plan_is_numbered():
    request = ModelRequest(purpose="plan", system="", messages=_history())
    text = asyncio.run(StubProvider().complete(request)).text
    assert text.startswith("1. ")
    assert "logs" in text


def test_get_provider_rejects_unknown():
    with pytest.raises(ValueError):
        get_provider(Settings(_env_file=None), "nope")


def _request():
    call = ToolCall(id="call_1", name="lookup_ticket", args={"ticket_id": "T-1"})
    return ModelRequest(
        purpose="act",
        system="be helpful",
        messages=[
            Message(role="user", content="hi"),
            Message(role="assistant", tool_calls=[call]),
            Message(role="tool", tool_call_id="call_1", content='{"ok": true}'),
        ],
        tools=[ToolSpec(name="lookup_ticket", description="d", parameters={"type": "object"})],
    )


def test_openai_message_shape():
    messages = to_openai_messages(_request())
    assert messages[0] == {"role": "system", "content": "be helpful"}
    assert messages[2]["tool_calls"][0]["function"]["name"] == "lookup_ticket"
    assert messages[3] == {"role": "tool", "tool_call_id": "call_1", "content": '{"ok": true}'}


def test_anthropic_message_shape():
    messages = to_anthropic_messages(_request())
    assert messages[1]["content"][0]["type"] == "tool_use"
    assert messages[2]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "call_1",
        "content": '{"ok": true}',
    }


def test_openai_provider_parses_tool_calls():
    call = SimpleNamespace(
        id="c1", function=SimpleNamespace(name="search_docs", arguments='{"query": "sso"}')
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[call]))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3),
    )

    class FakeCompletions:
        async def create(self, **kwargs):
            assert kwargs["tools"][0]["function"]["name"] == "lookup_ticket"
            return response

    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    result = asyncio.run(OpenAIProvider("gpt-test", client=client).complete(_request()))
    assert result.tool_calls[0].args == {"query": "sso"}
    assert result.usage == {"input_tokens": 10, "output_tokens": 3}


def test_anthropic_provider_parses_tool_use():
    blocks = [
        SimpleNamespace(type="text", text="Checking the docs."),
        SimpleNamespace(type="tool_use", id="tu_1", name="search_docs", input={"query": "sso"}),
    ]
    response = SimpleNamespace(
        content=blocks,
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=12, output_tokens=4),
    )

    class FakeMessages:
        async def create(self, **kwargs):
            assert kwargs["system"] == "be helpful"
            return response

    client = SimpleNamespace(messages=FakeMessages())
    result = asyncio.run(AnthropicProvider("claude-test", client=client).complete(_request()))
    assert result.text == "Checking the docs."
    assert result.tool_calls[0].id == "tu_1"


def test_missing_sdk_gives_install_hint(monkeypatch):
    import sys

    from relay.providers import MissingDependencyError

    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(MissingDependencyError, match=r"relay\[anthropic\]"):
        AnthropicProvider("claude-opus-5-5")


def test_anthropic_thinking_blocks_are_replayed_unchanged():
    """The assistant turn sent back on the next request is exactly what the API returned."""
    from relay.agent.nodes import AgentDeps, AgentNodes

    thinking = SimpleNamespace(type="thinking", thinking="", signature="sig-abc")
    tool_use = SimpleNamespace(
        type="tool_use", id="tu_1", name="lookup_ticket", input={"ticket_id": "T-1"}
    )
    first = SimpleNamespace(
        content=[thinking, tool_use],
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )
    second = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="Done.")],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=20, output_tokens=2),
    )
    sent = []

    class FakeMessages:
        def __init__(self):
            self.responses = [first, second]

        async def create(self, **kwargs):
            sent.append(kwargs)
            return self.responses.pop(0)

    provider = AnthropicProvider("claude-opus-5-5", client=SimpleNamespace(messages=FakeMessages()))

    class OneTool:
        async def list_tools(self):
            return [ToolSpec(name="lookup_ticket", description="d", parameters={"type": "object"})]

    nodes = AgentNodes(AgentDeps(provider, OneTool(), Settings(_env_file=None)))
    state = {"messages": [Message(role="user", content="ticket").model_dump()], "plan": []}
    update = asyncio.run(nodes.act(state))
    tool_result = Message(role="tool", tool_call_id="tu_1", content='{"ok": true}').model_dump()
    state["messages"] += [*update["messages"], tool_result]
    asyncio.run(nodes.act(state))

    replayed = sent[1]["messages"][1]
    assert replayed["role"] == "assistant"
    assert replayed["content"][0] == {"type": "thinking", "thinking": "", "signature": "sig-abc"}
    assert replayed["content"][1]["type"] == "tool_use"
    assert sent[0]["system"] == sent[1]["system"]  # the system prompt never changes mid-run
    assert sent[0]["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}


def test_anthropic_max_tokens_without_output_is_an_error():
    response = SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking="", signature="s")],
        stop_reason="max_tokens",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )

    class FakeMessages:
        async def create(self, **kwargs):
            assert kwargs["max_tokens"] >= 16_000
            return response

    provider = AnthropicProvider("claude-opus-5-5", client=SimpleNamespace(messages=FakeMessages()))
    from relay.providers import ProviderError

    with pytest.raises(ProviderError, match="max_tokens"):
        asyncio.run(provider.complete(_request()))
