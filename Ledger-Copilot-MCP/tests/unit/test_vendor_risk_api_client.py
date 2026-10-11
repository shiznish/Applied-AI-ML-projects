from __future__ import annotations

from typing import Any

import httpx
import pytest

from ledger_copilot.integrations.vendor_risk_api_client import (
    CircuitOpenError,
    VendorRiskApiClient,
    VendorRiskApiError,
    VendorRiskLevel,
)


class _FakeHttpClient:
    def __init__(self, responses: list[httpx.Response | Exception]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def get(self, *args: Any, **kwargs: Any) -> httpx.Response:
        self.calls += 1
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _response(status_code: int, json_body: dict[str, Any] | None = None) -> httpx.Response:
    return httpx.Response(
        status_code,
        json=json_body or {"results": []},
        request=httpx.Request("GET", "https://api.opensanctions.org/search/default"),
    )


def _make_client(
    responses: list[httpx.Response | Exception],
    **overrides: Any,
) -> tuple[VendorRiskApiClient, _FakeHttpClient]:
    fake_http_client = _FakeHttpClient(responses)
    sleeps: list[float] = []
    client = VendorRiskApiClient(
        base_url="https://api.opensanctions.org",
        http_client=fake_http_client,  # type: ignore[arg-type]
        max_retries=overrides.pop("max_retries", 3),
        backoff_base_seconds=0.0,
        sleep=sleeps.append,
        clock=overrides.pop("clock", lambda: 0.0),
        **overrides,
    )
    return client, fake_http_client


def test_get_risk_score_success_no_matches() -> None:
    client, fake_http_client = _make_client([_response(200, {"results": []})])

    assessment = client.get_risk_score("Acme Corp")

    assert assessment.vendor_name == "Acme Corp"
    assert assessment.match_count == 0
    assert assessment.top_match_score is None
    assert assessment.risk_level == VendorRiskLevel.LOW
    assert fake_http_client.calls == 1


def test_get_risk_score_high_risk_match() -> None:
    client, _ = _make_client(
        [_response(200, {"results": [{"score": 0.9}, {"score": 0.4}]})]
    )

    assessment = client.get_risk_score("Shady Vendor")

    assert assessment.match_count == 2
    assert assessment.top_match_score == 0.9
    assert assessment.risk_level == VendorRiskLevel.HIGH


def test_get_risk_score_raises_after_repeated_timeouts() -> None:
    client, fake_http_client = _make_client(
        [
            httpx.TimeoutException("timed out"),
            httpx.TimeoutException("timed out"),
            httpx.TimeoutException("timed out"),
        ],
        max_retries=3,
    )

    with pytest.raises(VendorRiskApiError):
        client.get_risk_score("Acme Corp")

    assert fake_http_client.calls == 3


def test_get_risk_score_retries_on_5xx_then_succeeds() -> None:
    client, fake_http_client = _make_client(
        [_response(503), _response(200, {"results": []})],
        max_retries=3,
    )

    assessment = client.get_risk_score("Acme Corp")

    assert assessment.risk_level == VendorRiskLevel.LOW
    assert fake_http_client.calls == 2


def test_circuit_opens_after_threshold_and_short_circuits() -> None:
    clock_time = {"value": 0.0}
    client, fake_http_client = _make_client(
        [
            httpx.TimeoutException("timed out"),
            httpx.TimeoutException("timed out"),
        ],
        max_retries=1,
        circuit_failure_threshold=2,
        circuit_cooldown_seconds=30.0,
        clock=lambda: clock_time["value"],
    )

    with pytest.raises(VendorRiskApiError):
        client.get_risk_score("Acme Corp")
    with pytest.raises(VendorRiskApiError):
        client.get_risk_score("Acme Corp")

    assert fake_http_client.calls == 2

    with pytest.raises(CircuitOpenError):
        client.get_risk_score("Acme Corp")

    assert fake_http_client.calls == 2
