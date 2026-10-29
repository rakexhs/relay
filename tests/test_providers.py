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
