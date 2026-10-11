"""OpenTelemetry setup for text2sql.

Sets up a `TracerProvider` (service name "text2sql"), exports spans to the
Jaeger instance already defined in docker-compose.yml, auto-instruments
FastAPI + asyncpg, and provides a couple of small helpers
(`traced_span`, `llm_call_span`) that the rest of the app uses to attach
project-specific and GenAI-convention-shaped span attributes.

Protocol choice — OTLP/HTTP, not gRPC
--------------------------------------
`opentelemetry-exporter-otlp` (the dependency already pinned in
pyproject.toml) bundles both the HTTP (`opentelemetry-exporter-otlp-proto-http`)
and gRPC (`...-proto-grpc`) exporters. This module uses the HTTP one, sent to
Jaeger's `COLLECTOR_OTLP_ENABLED` HTTP receiver on port 4318
(`http://localhost:4318/v1/traces`), for two reasons:
  1. It needs only a URL — no channel/credentials/keepalive setup — which
     keeps `configure_tracing()` simple and makes the "doesn't block startup
     when Jaeger isn't running" requirement easy to satisfy: constructing
     `OTLPSpanExporter` over HTTP just builds a `requests.Session`, it does
     not connect eagerly.
  2. Failures degrade predictably: `BatchSpanProcessor` calls `.export()` from
     its own background worker thread, so a connection-refused/timeout there
     is caught inside the SDK, logged, and the batch is dropped — it never
     raises into request-handling code or import time. (Confirmed by reading
     opentelemetry-sdk's `BatchSpanProcessor.export()` / `_worker` — the
     worker loop wraps the exporter call in try/except and never re-raises
     into the caller.)
If a project instead wanted gRPC (e.g. for a single long-lived connection to
a collocated collector), swap the import for
`opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter` and
point `TEXT2SQL_OTLP_ENDPOINT` at `localhost:4317` (no path, no scheme).

GenAI semantic-convention note
-------------------------------
Span/attribute names below (`gen_ai.generate_content`, `gen_ai.system`,
`gen_ai.request.model`, `gen_ai.usage.input_tokens` /
`gen_ai.usage.output_tokens`) follow OpenTelemetry's published GenAI semantic
conventions as of recent releases. The GenAI conventions are still marked
"development" upstream and have shifted names across versions, so treat this
as a reasonable, clearly-labeled approximation of the convention rather than
a claim of certified compliance against one pinned spec version.
"""
import contextvars
import logging
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from ..config import settings

logger = logging.getLogger("text2sql.observability")

SERVICE_NAME = "text2sql"
TRACER_NAME = "text2sql"

# Set once by configure_tracing() so tests/other modules can introspect what's
# active without threading the provider through every call site.
_tracer_provider: TracerProvider | None = None

# Per-request correlation id, propagated via contextvars rather than as an
# extra parameter threaded through every pipeline/llm function — asyncio
# copies the context into each request's Task, so setting this once in
# app/main.py's request handler (or app/text2sql/pipeline.py's run()) makes it
# visible to every span created later in that same request's call graph
# without changing any of those functions' signatures.
_correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "text2sql_correlation_id", default=""
)


def set_correlation_id(correlation_id: str):
    """Sets the current request's correlation id for span attribution.
    Returns a token suitable for `reset_correlation_id`."""
    return _correlation_id_var.set(correlation_id or "")


def reset_correlation_id(token) -> None:
    _correlation_id_var.reset(token)


def get_correlation_id() -> str:
    return _correlation_id_var.get()


def get_tracer() -> trace.Tracer:
    """Returns the text2sql tracer. Safe to call before configure_tracing() —
    the OTel API returns a working no-op tracer in that case, so importing
    modules and calling this at module scope never raises.

    Deliberately goes through our own `_tracer_provider` module variable
    rather than `trace.get_tracer(...)` (which reads OTel's own process-wide
    global provider) — the OTel global can only be set ONCE per process
    (`trace.set_tracer_provider` silently no-ops on a second call, with a
    logged warning), so whichever test module happens to import app.main
    first would permanently win the real OTLP exporter for every later test.
    Routing through our own variable lets every `configure_tracing()` call —
    including a test's in-memory-exporter one — actually take effect for our
    own hand-written spans, regardless of import order.
    """
    if _tracer_provider is not None:
        return _tracer_provider.get_tracer(TRACER_NAME)
    return trace.get_tracer(TRACER_NAME)


def configure_tracing(
    app=None,
    span_exporter: SpanExporter | None = None,
    endpoint: str | None = None,
) -> TracerProvider:
    """Sets up the global TracerProvider and (optionally) instruments a
    FastAPI app + asyncpg.

    Args:
        app: a FastAPI instance to instrument via FastAPIInstrumentor. Pass
            None to only set up the provider (e.g. in tests that just want to
            inspect spans from directly-called code, no ASGI app involved).
        span_exporter: override the exporter — this is the seam tests use to
            swap in `InMemorySpanExporter` instead of the real OTLP/HTTP one,
            so span assertions never need a live Jaeger, network, or thread
            timing games.
        endpoint: override the OTLP/HTTP endpoint URL. Defaults to
            `settings.otlp_endpoint` (env var TEXT2SQL_OTLP_ENDPOINT).

    Never raises just because Jaeger isn't reachable — exporter construction
    doesn't connect, and BatchSpanProcessor exports off the request path on a
    background thread.
    """
    global _tracer_provider

    resource = Resource.create({"service.name": SERVICE_NAME})
    provider = TracerProvider(resource=resource)

    if span_exporter is None:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            span_exporter = OTLPSpanExporter(endpoint=endpoint or settings.otlp_endpoint)
        except Exception:
            logger.warning(
                "Could not construct the OTLP/HTTP span exporter — tracing "
                "will run with no exporter attached (spans are created but "
                "not shipped anywhere).", exc_info=True,
            )
            span_exporter = None

    if span_exporter is not None:
        provider.add_span_processor(BatchSpanProcessor(span_exporter))

    trace.set_tracer_provider(provider)
    _tracer_provider = provider

    if app is not None:
        try:
            FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
        except Exception:
            logger.warning("FastAPI auto-instrumentation failed", exc_info=True)

    try:
        instrumentor = AsyncPGInstrumentor()
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument(tracer_provider=provider)
    except Exception:
        logger.warning("asyncpg auto-instrumentation failed", exc_info=True)

    return provider


def shutdown_tracing() -> None:
    """Flushes and shuts down the current provider's span processors. Best
    used at app shutdown so any spans still batched get a last chance to
    export (or, in tests, so exporter state is deterministic)."""
    if _tracer_provider is not None:
        _tracer_provider.shutdown()


@contextmanager
def traced_span(name: str, attributes: dict | None = None):
    """Starts a span with the standard text2sql.correlation_id attribute
    pre-populated from the contextvar, plus any extra attributes given.
    Yields the span so callers can add step-specific attributes once results
    are known."""
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        cid = get_correlation_id()
        if cid:
            span.set_attribute("text2sql.correlation_id", cid)
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        yield span


@contextmanager
def llm_call_span(operation: str, prompt_version: str | None = None):
    """Wraps one LLM provider call in a span following (an approximation of)
    OpenTelemetry's GenAI semantic conventions — see the module docstring for
    the compliance caveat. `operation` is a short name such as
    "generate_content", "back_translate", "judge_alignment", "repair", or
    "generate_alternative" — used both as part of the span name and as the
    `gen_ai.operation.name` attribute.

    `prompt_version` is the calling module's `PROMPT_VERSION` constant (see
    app/prompts/*.py) — set as a span attribute when given, so a trace (and
    anything that samples off it, e.g. the audit log) can be correlated with
    exactly which prompt revision produced the SQL/answer in question. Git
    history gives you the diff; this gives after-the-fact traceability
    without a runtime prompt-loading/versioning system.

    Token usage (gen_ai.usage.input_tokens / gen_ai.usage.output_tokens) is
    NOT set here — the provider call happens inside the `with` block in
    app/llm.py, which sets those attributes on `trace.get_current_span()`
    once it has the actual response object, since that's the only place the
    real usage numbers are available. See app/llm.py's module docstring for
    why they aren't threaded back up through _parse_structured's return value
    instead.
    """
    tracer = get_tracer()
    span_name = f"gen_ai.{operation}"
    with tracer.start_as_current_span(span_name) as span:
        span.set_attribute("gen_ai.system", settings.llm_provider)
        span.set_attribute("gen_ai.request.model", settings.model)
        span.set_attribute("gen_ai.operation.name", operation)
        if prompt_version:
            span.set_attribute("text2sql.prompt_version", prompt_version)
        cid = get_correlation_id()
        if cid:
            span.set_attribute("text2sql.correlation_id", cid)
        yield span


def set_llm_usage_attributes(input_tokens: int | None, output_tokens: int | None) -> None:
    """Sets GenAI token-usage attributes on the current span, if any is
    active and the values are available. Safe to call unconditionally — a
    no-op span (tracing not configured, or no span currently open) silently
    ignores the attribute writes."""
    span = trace.get_current_span()
    if input_tokens is not None:
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
    if output_tokens is not None:
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)


__all__ = [
    "configure_tracing",
    "shutdown_tracing",
    "get_tracer",
    "traced_span",
    "llm_call_span",
    "set_llm_usage_attributes",
    "set_correlation_id",
    "reset_correlation_id",
    "get_correlation_id",
    "ReadableSpan",
]
