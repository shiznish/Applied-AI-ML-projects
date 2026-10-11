from __future__ import annotations

import httpx2
import pytest
from fastmcp.utilities.asgi_transport import StreamingASGITransport

from ledger_copilot.audit.store import AuditStore
from ledger_copilot.server import create_app


@pytest.mark.asyncio
async def test_liveness_always_healthy() -> None:
    app = create_app()
    http_app = app.http_app()

    async with http_app.router.lifespan_context(http_app):
        transport = StreamingASGITransport(http_app)
        async with httpx2.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_readiness_healthy_when_audit_store_reachable() -> None:
    store = AuditStore()
    app = create_app(audit_store=store)
    http_app = app.http_app()

    async with http_app.router.lifespan_context(http_app):
        transport = StreamingASGITransport(http_app)
        async with httpx2.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/readyz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_readiness_unhealthy_when_audit_store_forced_to_fail() -> None:
    store = AuditStore()
    store._connection.close()
    app = create_app(audit_store=store)
    http_app = app.http_app()

    async with http_app.router.lifespan_context(http_app):
        transport = StreamingASGITransport(http_app)
        async with httpx2.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/readyz")

    assert response.status_code == 503
    assert response.json()["status"] == "unhealthy"
