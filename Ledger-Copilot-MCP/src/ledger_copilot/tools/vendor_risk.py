from __future__ import annotations

from typing import Literal

from fastmcp import FastMCP
from pydantic import BaseModel

from ledger_copilot.config import Settings
from ledger_copilot.integrations.vendor_risk_api_client import (
    VendorRiskApiClient,
    VendorRiskApiError,
    VendorRiskAssessment,
)

TOOL_SCOPE = "procurement:vendor:read"

_settings = Settings()
_vendor_risk_client = VendorRiskApiClient(
    base_url=_settings.vendor_risk_api_base_url,
    api_key=_settings.vendor_risk_api_key,
    timeout_seconds=_settings.vendor_risk_timeout_seconds,
    max_retries=_settings.vendor_risk_max_retries,
    circuit_failure_threshold=_settings.vendor_risk_circuit_failure_threshold,
    circuit_cooldown_seconds=_settings.vendor_risk_circuit_cooldown_seconds,
)


class GetVendorRiskScoreInput(BaseModel):
    vendor_name: str


class VendorRiskScoreFound(BaseModel):
    found: Literal[True] = True
    assessment: VendorRiskAssessment


class VendorRiskScoreUnavailable(BaseModel):
    found: Literal[False] = False
    vendor_name: str
    reason: str


GetVendorRiskScoreOutput = VendorRiskScoreFound | VendorRiskScoreUnavailable


def register(app: FastMCP) -> None:
    @app.tool(
        name="get_vendor_risk_score",
        description=(
            "Screen a vendor by name against the OpenSanctions sanctions/PEP "
            "database and return a risk assessment (match count, top match "
            "score, risk level), or an unavailable result if the screening "
            "service can't be reached."
        ),
        meta={"scope": TOOL_SCOPE},
    )
    def get_vendor_risk_score_tool(input: GetVendorRiskScoreInput) -> GetVendorRiskScoreOutput:
        try:
            assessment = _vendor_risk_client.get_risk_score(input.vendor_name)
        except VendorRiskApiError as exc:
            return VendorRiskScoreUnavailable(vendor_name=input.vendor_name, reason=str(exc))
        return VendorRiskScoreFound(assessment=assessment)
