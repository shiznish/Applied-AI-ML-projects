from __future__ import annotations

import httpx
import pytest
from fastmcp import Client

from ledger_copilot.audit.store import AuditStore
from ledger_copilot.server import create_app
from ledger_copilot.tools import vendor_risk


@pytest.fixture
def store() -> AuditStore:
    return AuditStore()


@pytest.fixture(autouse=True)
def _fast_vendor_risk_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the vendor-risk client at a mock transport so the audit test
    for `get_vendor_risk_score` doesn't depend on real network access."""

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    monkeypatch.setattr(
        vendor_risk._vendor_risk_client,
        "_http_client",
        httpx.Client(transport=httpx.MockTransport(_handler)),
    )


def _one(store: AuditStore, tool_name: str):  # type: ignore[no-untyped-def]
    records = store.query(tool_name=tool_name)
    assert len(records) == 1
    return records[0]


def _common_assertions(record, tool_name: str) -> None:  # type: ignore[no-untyped-def]
    assert record.tool_name == tool_name
    assert record.decision == "allowed"
    assert record.result_status == "success"
    assert record.latency_ms >= 0
    assert record.trace_id
    assert record.record_id
    # In-memory `fastmcp.Client` transport has no OAuth concept (see
    # middleware/auth.py's _ENFORCED_TRANSPORTS), so actor/scopes are absent.
    assert record.actor_user_id is None
    assert record.scopes_presented == []
    assert record.on_behalf_of_chain == []


@pytest.mark.asyncio
async def test_get_po_status_produces_audit_record(store: AuditStore) -> None:
    app = create_app(audit_store=store)
    async with Client(app) as client:
        await client.call_tool("get_po_status", {"input": {"po_id": "PO-1001"}})

    record = _one(store, "get_po_status")
    _common_assertions(record, "get_po_status")
    assert record.arguments == {"input": {"po_id": "PO-1001"}}
    assert record.idempotency_key is None
    assert record.approval_id is None


@pytest.mark.asyncio
async def test_get_vendor_risk_score_produces_audit_record(store: AuditStore) -> None:
    app = create_app(audit_store=store)
    async with Client(app) as client:
        await client.call_tool("get_vendor_risk_score", {"input": {"vendor_name": "Acme"}})

    record = _one(store, "get_vendor_risk_score")
    _common_assertions(record, "get_vendor_risk_score")
    assert record.arguments == {"input": {"vendor_name": "Acme"}}


@pytest.mark.asyncio
async def test_create_expense_request_hashes_sensitive_fields_keeps_amount(
    store: AuditStore,
) -> None:
    app = create_app(audit_store=store)
    args = {
        "input": {
            "idempotency_key": "audit-key-1",
            "requested_by": "alice",
            "cost_center": "CC-100",
            "amount": 42.5,
            "currency": "USD",
            "description": "team lunch",
        }
    }
    async with Client(app) as client:
        await client.call_tool("create_expense_request", args)

    record = _one(store, "create_expense_request")
    _common_assertions(record, "create_expense_request")
    assert record.idempotency_key == "audit-key-1"
    stored_input = record.arguments["input"]
    assert stored_input["amount"] == 42.5
    assert stored_input["currency"] == "USD"
    assert stored_input["cost_center"] == "CC-100"
    assert stored_input["requested_by"] != "alice"
    assert stored_input["requested_by"].startswith("sha256:")
    assert stored_input["description"] != "team lunch"
    assert stored_input["description"].startswith("sha256:")


@pytest.mark.asyncio
async def test_detect_invoice_anomalies_produces_audit_record(store: AuditStore) -> None:
    app = create_app(audit_store=store)
    args = {
        "input": {
            "vendor_name": "Acme",
            "invoices": [
                {"invoice_id": "INV-1", "amount": 100.0},
                {"invoice_id": "INV-2", "amount": 105.0},
                {"invoice_id": "INV-3", "amount": 9000.0},
            ],
        }
    }
    async with Client(app) as client:
        await client.call_tool("detect_invoice_anomalies", args)

    record = _one(store, "detect_invoice_anomalies")
    _common_assertions(record, "detect_invoice_anomalies")
    assert record.arguments["input"]["vendor_name"] == "Acme"
    assert record.arguments["input"]["invoices"][2]["amount"] == 9000.0


@pytest.mark.asyncio
async def test_generate_spend_report_produces_audit_record(store: AuditStore) -> None:
    app = create_app(audit_store=store)
    args = {"input": {"start_date": "2026-01-01", "end_date": "2026-03-31"}}
    async with Client(app) as client:
        await client.call_tool("generate_spend_report", args)

    record = _one(store, "generate_spend_report")
    _common_assertions(record, "generate_spend_report")
    assert record.arguments == args


@pytest.mark.asyncio
async def test_different_tool_calls_get_distinct_records(store: AuditStore) -> None:
    app = create_app(audit_store=store)
    async with Client(app) as client:
        await client.call_tool("get_po_status", {"input": {"po_id": "PO-1001"}})
        await client.call_tool("get_po_status", {"input": {"po_id": "PO-1002"}})

    records = store.query(tool_name="get_po_status")
    assert len(records) == 2
    assert records[0].record_id != records[1].record_id
    assert records[0].trace_id != records[1].trace_id
