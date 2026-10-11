from __future__ import annotations

from datetime import date

from fastmcp import Context, FastMCP
from pydantic import BaseModel

from ledger_copilot.domain.spend_report import (
    BudgetSummary,
    InvoiceSpendSummary,
    PurchaseOrderSpendSummary,
    summarize_budget,
    summarize_invoices,
    summarize_purchase_orders,
)
from ledger_copilot.integrations.erp_client import ErpClient

TOOL_SCOPE = "finance:report:read"

_erp_client = ErpClient()

_TOTAL_STAGES = 3


class GenerateSpendReportInput(BaseModel):
    start_date: date
    end_date: date
    cost_center: str | None = None


class GenerateSpendReportOutput(BaseModel):
    start_date: date
    end_date: date
    cost_center: str | None
    purchase_orders: PurchaseOrderSpendSummary
    invoices: InvoiceSpendSummary
    budget: BudgetSummary
    variance: float


def register(app: FastMCP) -> None:
    @app.tool(
        name="generate_spend_report",
        description=(
            "Aggregate purchase orders, invoices, and budget for a date "
            "range (optionally filtered by cost center) into a spend "
            "report with budget variance. Long-running; reports progress "
            "as each data source is aggregated."
        ),
        meta={"scope": TOOL_SCOPE},
    )
    async def generate_spend_report_tool(
        input: GenerateSpendReportInput, ctx: Context
    ) -> GenerateSpendReportOutput:
        await ctx.report_progress(0, _TOTAL_STAGES, "aggregating purchase orders")
        purchase_orders = summarize_purchase_orders(
            _erp_client.list_purchase_orders(), input.start_date, input.end_date, input.cost_center
        )

        await ctx.report_progress(1, _TOTAL_STAGES, "aggregating invoices")
        invoices = summarize_invoices(
            _erp_client.list_invoices(), input.start_date, input.end_date, input.cost_center
        )

        await ctx.report_progress(2, _TOTAL_STAGES, "aggregating budget")
        budget = summarize_budget(
            _erp_client.list_budgets(), input.start_date, input.end_date, input.cost_center
        )

        await ctx.report_progress(_TOTAL_STAGES, _TOTAL_STAGES, "spend report complete")
        return GenerateSpendReportOutput(
            start_date=input.start_date,
            end_date=input.end_date,
            cost_center=input.cost_center,
            purchase_orders=purchase_orders,
            invoices=invoices,
            budget=budget,
            variance=budget.budgeted_amount - invoices.total_invoice_amount,
        )
