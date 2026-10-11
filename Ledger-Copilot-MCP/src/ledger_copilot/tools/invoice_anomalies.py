from __future__ import annotations

from fastmcp import FastMCP
from pydantic import BaseModel

from ledger_copilot.domain.anomaly_detection import (
    Z_SCORE_THRESHOLD,
    InvoiceAmountRecord,
    InvoiceAnomalyScore,
    detect_anomalies,
)

TOOL_SCOPE = "procurement:invoice:read"


class DetectInvoiceAnomaliesInput(BaseModel):
    vendor_name: str
    invoices: list[InvoiceAmountRecord]
    threshold: float = Z_SCORE_THRESHOLD


class DetectInvoiceAnomaliesOutput(BaseModel):
    vendor_name: str
    scores: list[InvoiceAnomalyScore]


def register(app: FastMCP) -> None:
    @app.tool(
        name="detect_invoice_anomalies",
        description=(
            "Score a vendor's invoice amounts for statistical outliers using "
            "population z-score, flagging invoices whose amount deviates "
            "from that vendor's historical pattern beyond the given "
            "threshold."
        ),
        meta={"scope": TOOL_SCOPE},
    )
    def detect_invoice_anomalies_tool(
        input: DetectInvoiceAnomaliesInput,
    ) -> DetectInvoiceAnomaliesOutput:
        scores = detect_anomalies(input.invoices, threshold=input.threshold)
        return DetectInvoiceAnomaliesOutput(vendor_name=input.vendor_name, scores=scores)
