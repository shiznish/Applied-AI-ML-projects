from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel

from ledger_copilot.integrations.erp_client import ErpClient


class PurchaseOrderNotFoundError(Exception):
    def __init__(self, po_id: str) -> None:
        self.po_id = po_id
        super().__init__(f"purchase order not found: {po_id}")


class ReceiptStatus(StrEnum):
    NOT_RECEIVED = "not_received"
    PARTIALLY_RECEIVED = "partially_received"
    FULLY_RECEIVED = "fully_received"


class PurchaseOrderLineItem(BaseModel):
    line_number: int
    description: str
    quantity_ordered: float
    quantity_received: float
    unit_price: float


class PurchaseOrder(BaseModel):
    po_id: str
    vendor_name: str
    status: str
    order_date: date
    currency: str
    line_items: list[PurchaseOrderLineItem]
    receipt_status: ReceiptStatus


def _compute_receipt_status(line_items: list[PurchaseOrderLineItem]) -> ReceiptStatus:
    if all(item.quantity_received <= 0 for item in line_items):
        return ReceiptStatus.NOT_RECEIVED
    if all(item.quantity_received >= item.quantity_ordered for item in line_items):
        return ReceiptStatus.FULLY_RECEIVED
    return ReceiptStatus.PARTIALLY_RECEIVED


def get_po_status(po_id: str, erp_client: ErpClient) -> PurchaseOrder:
    record = erp_client.get_purchase_order(po_id)
    if record is None:
        raise PurchaseOrderNotFoundError(po_id)

    line_items = [PurchaseOrderLineItem.model_validate(item) for item in record["line_items"]]
    return PurchaseOrder(
        po_id=record["po_id"],
        vendor_name=record["vendor_name"],
        status=record["status"],
        order_date=record["order_date"],
        currency=record["currency"],
        line_items=line_items,
        receipt_status=_compute_receipt_status(line_items),
    )
