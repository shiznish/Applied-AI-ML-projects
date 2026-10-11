from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import httpx2
import pytest
from fastmcp import Client
from fastmcp.client.transports.http import StreamableHttpTransport
from fastmcp.exceptions import ToolError
from fastmcp.utilities.asgi_transport import StreamingASGITransport
from starlette.applications import Starlette

from tests.integration.conftest import TEST_ISSUER


@asynccontextmanager
async def _raw_client(http_app: Starlette) -> AsyncIterator[httpx2.AsyncClient]:
    async with http_app.router.lifespan_context(http_app):
        transport = StreamingASGITransport(http_app)
        async with httpx2.AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client

TOOL_SCOPES: dict[str, tuple[str, dict[str, object]]] = {
    "get_po_status": ("procurement:po:read", {"input": {"po_id": "PO-1"}}),
    "get_vendor_risk_score": ("procurement:vendor:read", {}),
    "create_expense_request": ("finance:expense:write", {}),
    "detect_invoice_anomalies": ("procurement:invoice:read", {}),
    "generate_spend_report": ("finance:report:read", {}),
}


def _client_for(
    http_app: Starlette,
    authorized_client_factory: Callable[..., Callable[..., object]],
    token: str,
) -> Client:
    return Client(
        StreamableHttpTransport(
            "http://testserver/mcp",
            auth=token,
            httpx_client_factory=authorized_client_factory(),
        )
    )


@pytest.mark.asyncio
async def test_protected_resource_metadata_endpoint(http_app) -> None:  # type: ignore[no-untyped-def]
    async with _raw_client(http_app) as raw_http_client:
        response = await raw_http_client.get("/.well-known/oauth-protected-resource/mcp")

    assert response.status_code == 200
    body = response.json()
    assert body["resource"] == "http://testserver/mcp"
    assert body["authorization_servers"] == [f"{TEST_ISSUER}/"]


@pytest.mark.asyncio
async def test_missing_token_returns_401_with_www_authenticate(http_app) -> None:  # type: ignore[no-untyped-def]
    async with _raw_client(http_app) as raw_http_client:
        response = await raw_http_client.post(
            "/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        )

    assert response.status_code == 401
    assert "Bearer" in response.headers["www-authenticate"]


@pytest.mark.asyncio
async def test_expired_token_returns_401(http_app, make_token) -> None:  # type: ignore[no-untyped-def]
    token = make_token(["procurement:po:read"], expires_in_seconds=-10)

    async with _raw_client(http_app) as raw_http_client:
        response = await raw_http_client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 401
    assert "Bearer" in response.headers["www-authenticate"]


@pytest.mark.asyncio
async def test_wrong_audience_returns_401(http_app, make_token) -> None:  # type: ignore[no-untyped-def]
    token = make_token(["procurement:po:read"], audience="some-other-resource")

    async with _raw_client(http_app) as raw_http_client:
        response = await raw_http_client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 401
    assert "Bearer" in response.headers["www-authenticate"]


@pytest.mark.asyncio
async def test_valid_token_valid_scope_passes(  # type: ignore[no-untyped-def]
    http_app, authorized_client_factory, make_token
) -> None:
    token = make_token(["procurement:po:read"])

    async with http_app.router.lifespan_context(http_app):
        async with _client_for(http_app, authorized_client_factory, token) as client:
            result = await client.call_tool("get_po_status", {"input": {"po_id": "PO-1"}})

    assert result.data.found is False
    assert result.data.po_id == "PO-1"


@pytest.mark.asyncio
async def test_valid_token_wrong_scope_denied(  # type: ignore[no-untyped-def]
    http_app, authorized_client_factory, make_token
) -> None:
    token = make_token(["some:unrelated:scope"])

    async with http_app.router.lifespan_context(http_app):
        async with _client_for(http_app, authorized_client_factory, token) as client:
            with pytest.raises(ToolError, match="insufficient scope"):
                await client.call_tool("get_po_status", {"input": {"po_id": "PO-1"}})


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", list(TOOL_SCOPES))
async def test_each_tool_denies_without_its_scope(
    http_app, authorized_client_factory, make_token, tool_name  # type: ignore[no-untyped-def]
) -> None:
    required_scope, args = TOOL_SCOPES[tool_name]
    token = make_token(["not-the-required-scope"])

    async with http_app.router.lifespan_context(http_app):
        async with _client_for(http_app, authorized_client_factory, token) as client:
            with pytest.raises(ToolError, match=f"required: {required_scope}"):
                await client.call_tool(tool_name, args)
