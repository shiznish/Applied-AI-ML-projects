from __future__ import annotations

import logging

from fastmcp.server.dependencies import get_http_headers
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools.base import ToolResult
from mcp_types import CallToolRequestParams
from opentelemetry import trace
from opentelemetry.propagate import extract

logger = logging.getLogger(__name__)

# Deferred to whatever TracerProvider server.py/a test installs via
# `trace.set_tracer_provider(...)` — a tracer obtained via `get_tracer()`
# before that call is a proxy that forwards to the real provider once set,
# per the OpenTelemetry API, so module import order doesn't matter here.
_tracer = trace.get_tracer("ledger_copilot")


class TracingMiddleware(Middleware):
    """Opens one OpenTelemetry span per tool call.

    Extracts the caller's W3C `traceparent` from the incoming HTTP request
    headers via `opentelemetry.propagate.extract` (the standard
    TextMapPropagator API) rather than hand-parsing the header, so the span
    becomes a child of the caller's trace when one was sent.
    `get_http_headers()` never raises and returns `{}` when there is no
    active HTTP request (stdio transport, the in-memory `fastmcp.Client`
    transport most tests use) — `extract({})` then yields the current
    (empty) context, so the span simply starts a new, unparented trace
    there, matching this codebase's general HTTP-only-context convention.
    """

    async def on_call_tool(
        self,
        context: MiddlewareContext[CallToolRequestParams],
        call_next: CallNext[CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        tool_name = context.message.name
        logger.info("tracing", extra={"tool": tool_name})

        parent_context = extract(get_http_headers())
        with _tracer.start_as_current_span(
            f"tool.{tool_name}",
            context=parent_context,
            kind=trace.SpanKind.SERVER,
        ) as span:
            span.set_attribute("mcp.tool.name", tool_name)
            return await call_next(context)
