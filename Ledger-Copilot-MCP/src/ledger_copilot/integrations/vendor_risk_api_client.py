"""Vendor screening via the OpenSanctions API — chosen over OpenCorporates
because it offers a usable free tier without a paid registry lookup quota,
and its sanctions/PEP match data is directly relevant to vendor risk
scoring rather than requiring a separate mapping from corporate filings."""

from __future__ import annotations

import time
from collections.abc import Callable
from enum import StrEnum
from typing import Any

import httpx
from pydantic import BaseModel

_HIGH_RISK_SCORE_THRESHOLD = 0.7


class VendorRiskApiError(Exception):
    pass


class CircuitOpenError(VendorRiskApiError):
    pass


class VendorRiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class VendorRiskAssessment(BaseModel):
    vendor_name: str
    match_count: int
    top_match_score: float | None
    risk_level: VendorRiskLevel


class _CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int,
        cooldown_seconds: float,
        clock: Callable[[], float],
    ) -> None:
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._clock = clock
        self._consecutive_failures = 0
        self._opened_at: float | None = None

    def before_call(self) -> None:
        if self._opened_at is None:
            return
        if self._clock() - self._opened_at < self._cooldown_seconds:
            raise CircuitOpenError("vendor risk API circuit is open")
        # Cooldown elapsed: let this call through as a half-open probe.

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._failure_threshold:
            self._opened_at = self._clock()


def _score_to_risk_level(top_match_score: float | None) -> VendorRiskLevel:
    if top_match_score is None:
        return VendorRiskLevel.LOW
    if top_match_score >= _HIGH_RISK_SCORE_THRESHOLD:
        return VendorRiskLevel.HIGH
    return VendorRiskLevel.MEDIUM


def _parse_assessment(vendor_name: str, payload: dict[str, Any]) -> VendorRiskAssessment:
    results: list[dict[str, Any]] = payload.get("results", [])
    scores = [float(result["score"]) for result in results if "score" in result]
    top_match_score = max(scores) if scores else None
    return VendorRiskAssessment(
        vendor_name=vendor_name,
        match_count=len(results),
        top_match_score=top_match_score,
        risk_level=_score_to_risk_level(top_match_score),
    )


class VendorRiskApiClient:
    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        http_client: httpx.Client | None = None,
        timeout_seconds: float = 5.0,
        max_retries: int = 3,
        backoff_base_seconds: float = 0.5,
        circuit_failure_threshold: int = 5,
        circuit_cooldown_seconds: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._http_client = http_client if http_client is not None else httpx.Client()
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._backoff_base_seconds = backoff_base_seconds
        self._sleep = sleep
        self._circuit_breaker = _CircuitBreaker(
            circuit_failure_threshold, circuit_cooldown_seconds, clock
        )

    def _headers(self) -> dict[str, str]:
        if self._api_key is None:
            return {}
        return {"Authorization": f"ApiKey {self._api_key}"}

    def get_risk_score(self, vendor_name: str) -> VendorRiskAssessment:
        self._circuit_breaker.before_call()

        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = self._http_client.get(
                    f"{self._base_url}/search/default",
                    params={"q": vendor_name},
                    headers=self._headers(),
                    timeout=self._timeout_seconds,
                )
            except httpx.TimeoutException as exc:
                last_error = exc
                self._circuit_breaker.record_failure()
            else:
                if response.status_code >= 500:
                    last_error = VendorRiskApiError(
                        f"vendor risk API returned {response.status_code}"
                    )
                    self._circuit_breaker.record_failure()
                else:
                    response.raise_for_status()
                    self._circuit_breaker.record_success()
                    return _parse_assessment(vendor_name, response.json())

            if attempt < self._max_retries - 1:
                self._sleep(self._backoff_base_seconds * (2**attempt))

        raise VendorRiskApiError(
            f"vendor risk lookup for {vendor_name!r} failed after {self._max_retries} attempts"
        ) from last_error
