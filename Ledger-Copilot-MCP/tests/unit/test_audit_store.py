from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import pytest

from ledger_copilot.audit.schema import AuditRecord
from ledger_copilot.audit.store import AuditStore


def _record(**overrides: object) -> AuditRecord:
    fields: dict[str, object] = {
        "record_id": "rec-1",
        "timestamp": datetime.now(UTC),
        "trace_id": "trace-1",
        "actor_user_id": "user-1",
        "on_behalf_of_chain": ["service-a"],
        "tool_name": "get_po_status",
        "arguments": {"input": {"po_id": "PO-1"}},
        "scopes_presented": ["procurement:po:read"],
        "decision": "allowed",
        "approval_id": None,
        "idempotency_key": None,
        "latency_ms": 12.5,
        "result_status": "success",
    }
    fields.update(overrides)
    return AuditRecord(**fields)  # type: ignore[arg-type]


def test_append_then_get_round_trips_all_fields() -> None:
    store = AuditStore()
    record = _record()

    store.append(record)
    fetched = store.get("rec-1")

    assert fetched == record


def test_get_missing_record_returns_none() -> None:
    store = AuditStore()

    assert store.get("does-not-exist") is None


def test_query_filters_by_tool_name() -> None:
    store = AuditStore()
    store.append(_record(record_id="rec-1", tool_name="get_po_status"))
    store.append(_record(record_id="rec-2", tool_name="create_expense_request"))

    results = store.query(tool_name="create_expense_request")

    assert [record.record_id for record in results] == ["rec-2"]


def test_query_without_filter_returns_all_records() -> None:
    store = AuditStore()
    store.append(_record(record_id="rec-1"))
    store.append(_record(record_id="rec-2"))

    results = store.query()

    assert {record.record_id for record in results} == {"rec-1", "rec-2"}


def test_update_is_rejected_by_the_append_only_trigger() -> None:
    store = AuditStore()
    store.append(_record())

    with pytest.raises(sqlite3.IntegrityError):
        store._connection.execute(
            "UPDATE audit_log SET tool_name = 'tampered' WHERE record_id = 'rec-1'"
        )


def test_delete_is_rejected_by_the_append_only_trigger() -> None:
    store = AuditStore()
    store.append(_record())

    with pytest.raises(sqlite3.IntegrityError):
        store._connection.execute("DELETE FROM audit_log WHERE record_id = 'rec-1'")

    assert store.get("rec-1") is not None
