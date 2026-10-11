from __future__ import annotations

from datetime import date

from ledger_copilot.domain.spend_report import (
    summarize_budget,
    summarize_invoices,
    summarize_purchase_orders,
)

PURCHASE_ORDERS = [
    {
        "po_id": "PO-1",
        "order_date": "2026-01-15",
        "cost_center": "CC-100",
        "line_items": [{"quantity_ordered": 2, "unit_price": 100.0}],
    },
    {
        "po_id": "PO-2",
        "order_date": "2026-04-01",
        "cost_center": "CC-100",
        "line_items": [{"quantity_ordered": 1, "unit_price": 50.0}],
    },
    {
        "po_id": "PO-3",
        "order_date": "2026-02-01",
        "cost_center": "CC-200",
        "line_items": [{"quantity_ordered": 1, "unit_price": 10.0}],
    },
]

INVOICES = [
    {"invoice_id": "INV-1", "invoice_date": "2026-01-20", "cost_center": "CC-100", "amount": 200.0},
    {"invoice_id": "INV-2", "invoice_date": "2026-05-01", "cost_center": "CC-100", "amount": 999.0},
    {"invoice_id": "INV-3", "invoice_date": "2026-02-10", "cost_center": "CC-200", "amount": 10.0},
]

BUDGETS = [
    {
        "cost_center": "CC-100",
        "period_start": "2026-01-01",
        "period_end": "2026-03-31",
        "budgeted_amount": 1000.0,
    },
    {
        "cost_center": "CC-200",
        "period_start": "2026-01-01",
        "period_end": "2026-03-31",
        "budgeted_amount": 500.0,
    },
]

Q1_START = date(2026, 1, 1)
Q1_END = date(2026, 3, 31)


def test_summarize_purchase_orders_filters_by_date_range_and_cost_center() -> None:
    summary = summarize_purchase_orders(PURCHASE_ORDERS, Q1_START, Q1_END, "CC-100")
    assert summary.po_count == 1
    assert summary.total_po_value == 200.0


def test_summarize_purchase_orders_no_cost_center_filter_includes_all_in_range() -> None:
    summary = summarize_purchase_orders(PURCHASE_ORDERS, Q1_START, Q1_END, None)
    assert summary.po_count == 2
    assert summary.total_po_value == 210.0


def test_summarize_invoices_filters_by_date_range_and_cost_center() -> None:
    summary = summarize_invoices(INVOICES, Q1_START, Q1_END, "CC-100")
    assert summary.invoice_count == 1
    assert summary.total_invoice_amount == 200.0


def test_summarize_invoices_no_cost_center_filter_includes_all_in_range() -> None:
    summary = summarize_invoices(INVOICES, Q1_START, Q1_END, None)
    assert summary.invoice_count == 2
    assert summary.total_invoice_amount == 210.0


def test_summarize_budget_sums_overlapping_periods_for_cost_center() -> None:
    summary = summarize_budget(BUDGETS, Q1_START, Q1_END, "CC-100")
    assert summary.budgeted_amount == 1000.0


def test_summarize_budget_no_cost_center_filter_sums_all_overlapping() -> None:
    summary = summarize_budget(BUDGETS, Q1_START, Q1_END, None)
    assert summary.budgeted_amount == 1500.0
