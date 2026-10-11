from __future__ import annotations

import json
from pathlib import Path

from ledger_copilot.integrations.erp_client import ErpClient


def _write_json(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text(json.dumps(records))
    return path


def _po(
    po_id: str = "PO-1",
    cost_center: str | None = "CC-100",
    line_items: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "po_id": po_id,
        "vendor_name": "Acme",
        "status": "open",
        "order_date": "2026-01-01",
        "currency": "USD",
        "cost_center": cost_center,
        "line_items": line_items
        or [
            {
                "line_number": 2,
                "description": "second",
                "quantity_ordered": 5,
                "quantity_received": 5,
                "unit_price": 20.0,
            },
            {
                "line_number": 1,
                "description": "first",
                "quantity_ordered": 10,
                "quantity_received": 10,
                "unit_price": 10.0,
            },
        ],
    }


def _invoice(invoice_id: str = "INV-1", match_discrepancy: bool = False) -> dict[str, object]:
    return {
        "invoice_id": invoice_id,
        "po_id": "PO-1",
        "vendor_name": "Acme",
        "created_by": "alice",
        "amount": 100.0,
        "currency": "USD",
        "match_discrepancy": match_discrepancy,
        "invoice_date": "2026-01-15",
        "cost_center": "CC-100",
    }


def _budget() -> dict[str, object]:
    return {
        "cost_center": "CC-100",
        "period_start": "2026-01-01",
        "period_end": "2026-03-31",
        "budgeted_amount": 10000.0,
    }


def _client(tmp_path: Path, *, orders=None, invoices=None, budgets=None) -> ErpClient:
    return ErpClient(
        fixture_path=_write_json(tmp_path / "pos.json", orders if orders is not None else [_po()]),
        invoice_fixture_path=_write_json(
            tmp_path / "invoices.json", invoices if invoices is not None else [_invoice()]
        ),
        budget_fixture_path=_write_json(
            tmp_path / "budgets.json", budgets if budgets is not None else [_budget()]
        ),
    )


def test_get_purchase_order_round_trips_with_line_items_ordered(tmp_path: Path) -> None:
    client = _client(tmp_path)

    record = client.get_purchase_order("PO-1")

    assert record is not None
    assert record["po_id"] == "PO-1"
    assert record["vendor_name"] == "Acme"
    assert record["cost_center"] == "CC-100"
    assert [item["line_number"] for item in record["line_items"]] == [1, 2]
    assert record["line_items"][0]["description"] == "first"


def test_get_purchase_order_missing_returns_none(tmp_path: Path) -> None:
    client = _client(tmp_path)

    assert client.get_purchase_order("PO-404") is None


def test_list_purchase_orders_returns_all_seeded_records(tmp_path: Path) -> None:
    client = _client(tmp_path, orders=[_po("PO-1"), _po("PO-2", cost_center="CC-200")])

    records = client.list_purchase_orders()

    assert {record["po_id"] for record in records} == {"PO-1", "PO-2"}
    assert all(len(record["line_items"]) == 2 for record in records)


def test_list_invoices_returns_all_seeded_records(tmp_path: Path) -> None:
    client = _client(
        tmp_path, invoices=[_invoice("INV-1"), _invoice("INV-2", match_discrepancy=True)]
    )

    records = client.list_invoices()

    assert {record["invoice_id"] for record in records} == {"INV-1", "INV-2"}


def test_list_budgets_returns_all_seeded_records(tmp_path: Path) -> None:
    client = _client(tmp_path, budgets=[_budget()])

    records = client.list_budgets()

    assert records == [_budget()]
