from __future__ import annotations

from typing import Literal

from fastmcp import FastMCP
from pydantic import BaseModel

from ledger_copilot.domain.procurement import (
    PurchaseOrder,
    PurchaseOrderNotFoundError,
    get_po_status,
)
from ledger_copilot.integrations.erp_client import ErpClient

TOOL_SCOPE = "procurement:po:read"

_erp_client = ErpClient()


class GetPoStatusInput(BaseModel):
    po_id: str


class PoStatusFound(BaseModel):
    found: Literal[True] = True
    purchase_order: PurchaseOrder


class PoStatusNotFound(BaseModel):
    found: Literal[False] = False
    po_id: str


GetPoStatusOutput = PoStatusFound | PoStatusNotFound


def register(app: FastMCP) -> None:
    @app.tool(
        name="get_po_status",
        description=(
            "Look up a purchase order by id and return its current status, "
            "vendor, line items, and computed receipt status (not/partially/"
            "fully received)."
        ),
        meta={"scope": TOOL_SCOPE},
    )
    def get_po_status_tool(input: GetPoStatusInput) -> GetPoStatusOutput:
        try:
            purchase_order = get_po_status(input.po_id, _erp_client)
        except PurchaseOrderNotFoundError:
            return PoStatusNotFound(po_id=input.po_id)
        return PoStatusFound(purchase_order=purchase_order)
