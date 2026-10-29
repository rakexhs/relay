"""OpenTelemetry spans for tool and model calls, plus optional LangSmith tracing.

- RELAY_OTEL_EXPORTER=console prints spans; =otlp ships them to OTEL_EXPORTER_OTLP_ENDPOINT
  (needs the `otlp` extra). With `none` the API calls are no-ops.
- LANGSMITH_TRACING=true (with the `langsmith` extra and LANGSMITH_API_KEY) also sends model
  calls to LangSmith. LangGraph picks the same variables up for graph-level traces.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from relay.config import Settings

_provider: trace.TracerProvider | None = None


def configure(provider: trace.TracerProvider | None) -> None:
    """Use a specific tracer provider (tests pass one with an in-memory exporter)."""
    global _provider
    _provider = provider


def setup_tracing(settings: Settings) -> None:
    if settings.otel_exporter == "none":
        return
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    provider = TracerProvider(resource=Resource.create({"service.name": settings.service_name}))
    if settings.otel_exporter == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        except ImportError as exc:
            from relay.providers.base import MissingDependencyError

            raise MissingDependencyError("opentelemetry-exporter-otlp-proto-http", "otlp") from exc
        exporter = OTLPSpanExporter()
    else:
        exporter = ConsoleSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    configure(provider)


def tracer() -> trace.Tracer:
    return trace.get_tracer("relay", tracer_provider=_provider)


def _short(value: Any, limit: int = 512) -> str:
    text = json.dumps(value, default=str, sort_keys=True)
    return text if len(text) <= limit else text[: limit - 3] + "..."


@contextmanager
def tool_span(name: str, args: dict[str, Any], run_id: str | None) -> Iterator[Span]:
    with tracer().start_as_current_span(f"tool {name}") as span:
        span.set_attribute("relay.tool.name", name)
        span.set_attribute("relay.tool.args", _short(args))
        if run_id:
            span.set_attribute("relay.run_id", run_id)
        yield span


def record_tool_result(span: Span, result: dict[str, Any], approved: bool) -> None:
    span.set_attribute("relay.tool.approved", approved)
    error = result.get("error") if isinstance(result, dict) else None
    if isinstance(error, dict):
        code = str(error.get("code"))
        span.set_attribute("relay.tool.error_code", code)
        if error.get("field"):
            span.set_attribute("relay.tool.error_field", str(error["field"]))
        span.set_status(Status(StatusCode.ERROR, code))


@contextmanager
def model_span(provider: str, model: str, purpose: str) -> Iterator[Span]:
    with tracer().start_as_current_span(f"model {purpose}") as span:
        span.set_attribute("relay.model.provider", provider)
        span.set_attribute("relay.model.name", model)
        span.set_attribute("relay.model.purpose", purpose)
        yield span


def langsmith_enabled() -> bool:
    return os.environ.get("LANGSMITH_TRACING", "").lower() in {"1", "true", "yes"}


def maybe_traceable(fn: Callable[..., Any], *, name: str, run_type: str) -> Callable[..., Any]:
    """Wrap fn with langsmith.traceable when LangSmith tracing is switched on and installed."""
    if not langsmith_enabled():
        return fn
    try:
        from langsmith import traceable
    except ImportError:
        return fn
    return traceable(name=name, run_type=run_type)(fn)
