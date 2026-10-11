from __future__ import annotations

from fastmcp import FastMCP

PROMPT_NAME = "quarterly-review"


def register(app: FastMCP) -> None:
    @app.prompt(name=PROMPT_NAME)
    def quarterly_review(cost_center: str) -> str:
        return (
            f"You are preparing a quarterly spend review for cost center "
            f"{cost_center}. Use generate_spend_report to pull purchase order, "
            f"invoice, and budget totals for the quarter, call out any budget "
            f"variance, and use detect_invoice_anomalies to flag invoices from "
            f"this cost center's vendors that warrant a closer look."
        )
