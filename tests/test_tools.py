import asyncio

import pytest

from relay.tools import CallMeta, ToolRegistry
from relay.tools.errors import ToolError
from relay.tools.logquery import parse_query, to_opensearch

from .conftest import make_world


def call(registry, name, args, **meta):
    return asyncio.run(registry.invoke(name, args, CallMeta(**meta)))


def test_six_tools_with_json_schemas(registry):
    specs = {spec.name: spec for spec in registry.specs()}
    assert set(specs) == {
        "search_docs",
        "lookup_ticket",
        "query_logs",
        "draft_reply",
        "escalate_to_human",
        "recall_customer_history",
    }
    assert specs["query_logs"].parameters["required"] == ["query"]
    assert "status_code" in specs["query_logs"].description


def test_invalid_args_return_structured_error(registry):
    result = call(registry, "lookup_ticket", {"ticket_id": "1042"})
    assert result["error"]["code"] == "invalid_arguments"
    assert result["error"]["field"] == "ticket_id"


def test_extra_args_are_rejected(registry):
    result = call(registry, "search_docs", {"query": "sso", "filters": {"x": 1}})
    assert result["error"]["code"] == "invalid_arguments"
    assert result["error"]["field"] == "filters"


def test_unknown_tool(registry):
    result = call(registry, "delete_account", {})
    assert result["error"]["code"] == "unknown_tool"
    assert "search_docs" in result["error"]["allowed"]


def test_malformed_log_query_names_field_and_allowed_values(registry):
    result = call(registry, "query_logs", {"query": "service:webhooks AND status:502"})
    error = result["error"]
    assert error["code"] == "invalid_query"
    assert error["field"] == "query"
    assert "status_code" in error["allowed"]
    assert "Unknown field 'status'" in error["hint"]


def test_relative_since_is_rejected_with_hint(registry):
    result = call(registry, "query_logs", {"query": "level:error", "since": "yesterday"})
    assert result["error"]["field"] == "since"
    assert "15m" in result["error"]["hint"]


def test_query_logs_filters_by_window_and_clauses(registry):
    result = call(
        registry, "query_logs", {"query": "service:webhooks AND status_code:502", "since": "24h"}
    )
    assert result["count"] == 1
    assert result["hits"][0]["request_id"] == "req_a1"


def test_log_backend_outage_is_retryable():
    registry = ToolRegistry(make_world(logs_down=True).context())
    result = call(registry, "query_logs", {"query": "level:error"})
    assert result["error"]["code"] == "backend_unavailable"
    assert result["error"]["retryable"] is True


def test_sending_needs_approval_but_drafting_does_not(registry, world):
    body = "Hi Dana, here is how to fix the webhook endpoint."
    blocked = call(registry, "draft_reply", {"ticket_id": "T-100", "body": body, "send": True})
    assert blocked["error"]["code"] == "approval_required"
    assert world.outbox.sent == []

    drafted = call(registry, "draft_reply", {"ticket_id": "T-100", "body": body})
    assert drafted["status"] == "drafted"

    sent = call(
        registry, "draft_reply", {"ticket_id": "T-100", "body": body, "send": True}, approved=True
    )
    assert sent["status"] == "sent"
    assert world.outbox.sent[0]["body"] == body


def test_escalation_records_reason_and_trajectory(registry, world):
    trajectory = [{"step": 1, "tool": "search_docs"}]
    result = call(
        registry,
        "escalate_to_human",
        {"ticket_id": "T-100", "reason": "needs_refund", "summary": "Customer wants a refund."},
        approved=True,
        run_id="run-1",
        trajectory=trajectory,
    )
    assert result["reason"] == "needs_refund"
    assert world.escalations.items[0]["trajectory"] == trajectory


def test_bad_reason_code_is_a_validation_error(registry):
    result = call(
        registry,
        "escalate_to_human",
        {"ticket_id": "T-100", "reason": "angry", "summary": "Customer is upset about this."},
        approved=True,
    )
    assert result["error"]["field"] == "reason"


def test_recall_ranks_relevant_memory_first(registry):
    result = call(
        registry, "recall_customer_history", {"customer_id": "C-7", "query": "webhook failures"}
    )
    assert result["memories"][0]["ticket_id"] == "T-050"


def test_search_docs_finds_webhook_article(registry):
    result = call(registry, "search_docs", {"query": "webhook 502 failing endpoint"})
    assert result["results"][0]["doc_id"] == "KB-101"
    assert result["results"][0]["score"] > 0.5


def test_lookup_ticket_not_found(registry):
    result = call(registry, "lookup_ticket", {"ticket_id": "T-999"})
    assert result["error"]["code"] == "not_found"


@pytest.mark.parametrize(
    "text, hint",
    [
        ("service:api OR service:auth", "OR is not supported"),
        ("service:api level:error", "Missing AND"),
        ("service:api AND", "ends with AND"),
        ('message:"unclosed', "unclosed quote"),
        ("status_code:5xx", "3-digit"),
    ],
)
def test_parse_query_errors(text, hint):
    with pytest.raises(ToolError) as excinfo:
        parse_query(text)
    assert hint in excinfo.value.hint


def test_parse_query_supports_not_and_quotes():
    query = parse_query('service:api AND NOT level:info AND message:"timed out"')
    assert [c.field for c in query.clauses] == ["service", "level", "message"]
    assert query.clauses[1].negate
    assert query.matches({"service": "api", "level": "error", "message": "request timed out"})


def test_to_opensearch_builds_bool_query():
    body = to_opensearch(parse_query("service:api AND NOT status_code:200"), "6h", 10)
    bool_query = body["query"]["bool"]
    assert {"term": {"service.keyword": "api"}} in bool_query["must"]
    assert bool_query["must_not"] == [{"term": {"status_code": 200}}]
    assert body["size"] == 10


def test_opensearch_backend_sends_dsl_and_maps_hits():
    from datetime import timedelta

    from relay.tools.backends import OpenSearchLogs

    class FakeClient:
        def __init__(self):
            self.calls = []

        def search(self, index, body):
            self.calls.append((index, body))
            source = {"@timestamp": "2026-03-02T11:00:00Z", "service": "api", "status_code": 500}
            return {"hits": {"hits": [{"_source": source}]}}

    client = FakeClient()
    backend = OpenSearchLogs("http://unused", "app-logs", client=client)
    hits = backend.search(parse_query("service:api"), timedelta(hours=6), 5, now=None)
    index, body = client.calls[0]
    assert index == "app-logs"
    assert body["query"]["bool"]["must"][0] == {"range": {"@timestamp": {"gte": "now-360m"}}}
    assert hits == [{"timestamp": "2026-03-02T11:00:00Z", "service": "api", "status_code": 500}]


def test_opensearch_errors_become_retryable_tool_errors():
    from datetime import timedelta

    from relay.tools.backends import OpenSearchLogs

    class DownClient:
        def search(self, index, body):
            raise ConnectionError("refused")

    backend = OpenSearchLogs("http://unused", "app-logs", client=DownClient())
    with pytest.raises(ToolError) as excinfo:
        backend.search(parse_query("level:error"), timedelta(hours=1), 5, now=None)
    assert excinfo.value.code == "backend_unavailable"
    assert excinfo.value.retryable
