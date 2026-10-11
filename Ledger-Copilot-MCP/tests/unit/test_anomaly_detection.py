from __future__ import annotations

from ledger_copilot.domain.anomaly_detection import (
    InvoiceAmountRecord,
    detect_anomalies,
)


def _records(amounts: list[float]) -> list[InvoiceAmountRecord]:
    return [
        InvoiceAmountRecord(invoice_id=f"INV-{index}", amount=amount)
        for index, amount in enumerate(amounts)
    ]


def test_planted_outlier_is_flagged_normal_invoices_are_not() -> None:
    amounts = [980.0, 1010.0, 1000.0, 990.0, 1005.0, 995.0, 1020.0, 50000.0]
    records = _records(amounts)

    scores = detect_anomalies(records)

    by_id = {score.invoice_id: score for score in scores}
    assert by_id["INV-7"].is_anomaly is True
    for invoice_id in ["INV-0", "INV-1", "INV-2", "INV-3", "INV-4", "INV-5", "INV-6"]:
        assert by_id[invoice_id].is_anomaly is False


def test_no_outlier_when_amounts_are_uniform() -> None:
    records = _records([500.0, 500.0, 500.0, 500.0])

    scores = detect_anomalies(records)

    assert all(score.is_anomaly is False for score in scores)
    assert all(score.z_score == 0.0 for score in scores)


def test_fewer_than_two_records_never_flagged() -> None:
    records = _records([1_000_000.0])

    scores = detect_anomalies(records)

    assert scores[0].is_anomaly is False
    assert scores[0].z_score == 0.0


def test_empty_records_returns_empty_list() -> None:
    assert detect_anomalies([]) == []


def test_custom_threshold_changes_sensitivity() -> None:
    amounts = [1000.0, 1010.0, 990.0, 1005.0, 1300.0]
    records = _records(amounts)

    lenient = detect_anomalies(records, threshold=5.0)
    strict = detect_anomalies(records, threshold=0.5)

    assert all(score.is_anomaly is False for score in lenient)
    assert any(score.is_anomaly is True for score in strict)
