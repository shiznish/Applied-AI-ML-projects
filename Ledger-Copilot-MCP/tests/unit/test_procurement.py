from __future__ import annotations

import json
from pathlib import Path

import pytest

from ledger_copilot.domain.procurement import (
    PurchaseOrderNotFoundError,
    ReceiptStatus,
    get_po_status,
)
from ledger_copilot.integrations.erp_client import ErpClient


def _write_fixture(tmp_path: Path, records: list[dict[str, object]]) -> ErpClient:
    fixture_path = tmp_path / "purchase_orders.json"
    fixture_path.write_text(json.dumps(records))
    return ErpClient(fixture_path=fixture_path)


def _line_item(
    quantity_ordered: float, quantity_received: float, line_number: int = 1
) -> dict[str, object]:
    return {
        "line_number": line_number,
        "description": "widget",
        "quantity_ordered": quantity_ordered,
        "quantity_received": quantity_received,
        "unit_price": 10.0,
    }


def test_get_po_status_fully_received(tmp_path: Path) -> None:
    erp_client = _write_fixture(
        tmp_path,
        [
            {
                "po_id": "PO-1",
                "vendor_name": "Acme",
                "status": "open",
                "order_date": "2026-01-01",
                "currency": "USD",
                "line_items": [_line_item(10, 10)],
            }
        ],
    )

    purchase_order = get_po_status("PO-1", erp_client)

    assert purchase_order.po_id == "PO-1"
    assert purchase_order.receipt_status == ReceiptStatus.FULLY_RECEIVED


def test_get_po_status_partially_received(tmp_path: Path) -> None:
    erp_client = _write_fixture(
        tmp_path,
        [
            {
                "po_id": "PO-2",
                "vendor_name": "Acme",
                "status": "open",
                "order_date": "2026-01-01",
                "currency": "USD",
                "line_items": [_line_item(10, 5, line_number=1), _line_item(4, 4, line_number=2)],
            }
        ],
    )

    purchase_order = get_po_status("PO-2", erp_client)

    assert purchase_order.receipt_status == ReceiptStatus.PARTIALLY_RECEIVED


def test_get_po_status_not_received(tmp_path: Path) -> None:
    erp_client = _write_fixture(
        tmp_path,
        [
            {
                "po_id": "PO-3",
                "vendor_name": "Acme",
                "status": "open",
                "order_date": "2026-01-01",
                "currency": "USD",
                "line_items": [_line_item(10, 0)],
            }
        ],
    )

    purchase_order = get_po_status("PO-3", erp_client)

    assert purchase_order.receipt_status == ReceiptStatus.NOT_RECEIVED


def test_get_po_status_raises_when_not_found(tmp_path: Path) -> None:
    erp_client = _write_fixture(tmp_path, [])

    with pytest.raises(PurchaseOrderNotFoundError) as exc_info:
        get_po_status("PO-404", erp_client)

    assert exc_info.value.po_id == "PO-404"
