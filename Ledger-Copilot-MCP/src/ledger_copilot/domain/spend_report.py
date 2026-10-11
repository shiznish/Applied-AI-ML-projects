from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel


class PurchaseOrderSpendSummary(BaseModel):
    po_count: int
    total_po_value: float


class InvoiceSpendSummary(BaseModel):
    invoice_count: int
    total_invoice_amount: float


class BudgetSummary(BaseModel):
    budgeted_amount: float


def _overlaps(record_start: date, record_end: date, start_date: date, end_date: date) -> bool:
    return record_start <= end_date and record_end >= start_date


def _matches_cost_center(record: dict[str, Any], cost_center: str | None) -> bool:
    return cost_center is None or record.get("cost_center") == cost_center


def summarize_purchase_orders(
    records: list[dict[str, Any]],
    start_date: date,
    end_date: date,
    cost_center: str | None,
) -> PurchaseOrderSpendSummary:
    total = 0.0
    count = 0
    for record in records:
        order_date = date.fromisoformat(record["order_date"])
        if not (start_date <= order_date <= end_date):
            continue
        if not _matches_cost_center(record, cost_center):
            continue
        count += 1
        total += sum(
            item["quantity_ordered"] * item["unit_price"] for item in record["line_items"]
        )
    return PurchaseOrderSpendSummary(po_count=count, total_po_value=total)


def summarize_invoices(
    records: list[dict[str, Any]],
    start_date: date,
    end_date: date,
    cost_center: str | None,
) -> InvoiceSpendSummary:
    total = 0.0
    count = 0
    for record in records:
        invoice_date = date.fromisoformat(record["invoice_date"])
        if not (start_date <= invoice_date <= end_date):
            continue
        if not _matches_cost_center(record, cost_center):
            continue
        count += 1
        total += record["amount"]
    return InvoiceSpendSummary(invoice_count=count, total_invoice_amount=total)


def summarize_budget(
    records: list[dict[str, Any]],
    start_date: date,
    end_date: date,
    cost_center: str | None,
) -> BudgetSummary:
    total = 0.0
    for record in records:
        period_start = date.fromisoformat(record["period_start"])
        period_end = date.fromisoformat(record["period_end"])
        if not _overlaps(period_start, period_end, start_date, end_date):
            continue
        if not _matches_cost_center(record, cost_center):
            continue
        total += record["budgeted_amount"]
    return BudgetSummary(budgeted_amount=total)
