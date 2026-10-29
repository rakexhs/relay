import asyncio

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from relay import tracing
from relay.tools import CallMeta


def test_each_tool_call_gets_a_span(registry):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracing.configure(provider)
    try:
        asyncio.run(registry.invoke("search_docs", {"query": "sso"}, CallMeta(run_id="run-t")))
        asyncio.run(
            registry.invoke("query_logs", {"query": "status:500"}, CallMeta(run_id="run-t"))
        )
    finally:
        tracing.configure(None)
    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert set(spans) == {"tool search_docs", "tool query_logs"}
    assert spans["tool search_docs"].attributes["relay.run_id"] == "run-t"
    failed = spans["tool query_logs"]
    assert failed.status.status_code == StatusCode.ERROR
    assert failed.attributes["relay.tool.error_code"] == "invalid_query"
    assert failed.attributes["relay.tool.error_field"] == "query"


def test_langsmith_wrapper_is_a_no_op_when_disabled(monkeypatch):
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)

    def fn():
        return 1

    assert tracing.maybe_traceable(fn, name="x", run_type="llm") is fn
