from __future__ import annotations

import pytest
from fastmcp import Client

from ledger_copilot.server import create_app


def _report_args(cost_center: str | None = None) -> dict[str, object]:
    input_args: dict[str, object] = {"start_date": "2026-01-01", "end_date": "2026-03-31"}
    if cost_center is not None:
        input_args["cost_center"] = cost_center
    return {"input": input_args}


@pytest.mark.asyncio
async def test_generate_spend_report_emits_progress_before_final_result() -> None:
    app = create_app()
    progress_updates: list[tuple[float, float | None, str | None]] = []

    async def on_progress(progress: float, total: float | None, message: str | None) -> None:
        progress_updates.append((progress, total, message))

    async with Client(app, progress_handler=on_progress) as client:
        result = await client.call_tool("generate_spend_report", _report_args())

    assert len(progress_updates) >= 2
    assert result.data.purchase_orders.po_count == 3
    assert result.data.invoices.invoice_count == 3


@pytest.mark.asyncio
async def test_generate_spend_report_result_matches_fixtures_for_full_range() -> None:
    app = create_app()
    async with Client(app) as client:
        result = await client.call_tool("generate_spend_report", _report_args())

    assert result.data.purchase_orders.po_count == 3
    assert result.data.purchase_orders.total_po_value == 14010.0
    assert result.data.invoices.invoice_count == 3
    assert result.data.invoices.total_invoice_amount == 65400.0
    assert result.data.budget.budgeted_amount == 60000.0
    assert result.data.variance == -5400.0


@pytest.mark.asyncio
async def test_generate_spend_report_filters_by_cost_center() -> None:
    app = create_app()
    async with Client(app) as client:
        result = await client.call_tool("generate_spend_report", _report_args("CC-100"))

    assert result.data.purchase_orders.po_count == 2
    assert result.data.purchase_orders.total_po_value == 9210.0
    assert result.data.invoices.invoice_count == 2
    assert result.data.invoices.total_invoice_amount == 5400.0
    assert result.data.budget.budgeted_amount == 10000.0
    assert result.data.variance == 4600.0
