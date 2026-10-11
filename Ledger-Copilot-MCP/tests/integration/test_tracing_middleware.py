from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest
from fastmcp import Client
from fastmcp.client.transports.http import StreamableHttpTransport
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from starlette.applications import Starlette

_TRACE_ID_HEX = "4bf92f3577b34da6a3ce929d0e0e4736"
_TRACEPARENT = f"00-{_TRACE_ID_HEX}-00f067aa0ba902b7-01"


@pytest.fixture
def span_exporter() -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    previous_provider = trace.get_tracer_provider()
    trace.set_tracer_provider(provider)
    yield exporter
    trace.set_tracer_provider(previous_provider)


@pytest.mark.asyncio
async def test_tool_call_span_is_child_of_incoming_traceparent(
    http_app: Starlette,
    authorized_client_factory: Callable[..., Callable[..., object]],
    make_token: Callable[..., str],
    span_exporter: InMemorySpanExporter,
) -> None:
    token = make_token(["procurement:po:read"])

    async with http_app.router.lifespan_context(http_app):
        client = Client(
            StreamableHttpTransport(
                "http://testserver/mcp",
                auth=token,
                headers={"traceparent": _TRACEPARENT},
                httpx_client_factory=authorized_client_factory(),
            )
        )
        async with client:
            result = await client.call_tool("get_po_status", {"input": {"po_id": "PO-1"}})

    assert result.data.po_id == "PO-1"

    spans = span_exporter.get_finished_spans()
    tool_spans = [span for span in spans if span.name == "tool.get_po_status"]
    assert len(tool_spans) == 1
    assert format(tool_spans[0].context.trace_id, "032x") == _TRACE_ID_HEX
