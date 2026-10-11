from __future__ import annotations

import statistics

from pydantic import BaseModel

# Z-score over isolation-forest: univariate amount outlier detection on a
# single vendor's invoice history needs no multivariate model, stays
# explainable to an auditor (the z-score itself is the justification), and
# needs no new ML dependency.
Z_SCORE_THRESHOLD = 2.5


class InvoiceAmountRecord(BaseModel):
    invoice_id: str
    amount: float


class InvoiceAnomalyScore(BaseModel):
    invoice_id: str
    amount: float
    z_score: float
    is_anomaly: bool


def detect_anomalies(
    records: list[InvoiceAmountRecord],
    threshold: float = Z_SCORE_THRESHOLD,
) -> list[InvoiceAnomalyScore]:
    if len(records) < 2:
        return [
            InvoiceAnomalyScore(
                invoice_id=record.invoice_id,
                amount=record.amount,
                z_score=0.0,
                is_anomaly=False,
            )
            for record in records
        ]

    amounts = [record.amount for record in records]
    mean = statistics.fmean(amounts)
    stdev = statistics.pstdev(amounts, mu=mean)

    if stdev == 0:
        return [
            InvoiceAnomalyScore(
                invoice_id=record.invoice_id,
                amount=record.amount,
                z_score=0.0,
                is_anomaly=False,
            )
            for record in records
        ]

    scores = []
    for record in records:
        z_score = (record.amount - mean) / stdev
        scores.append(
            InvoiceAnomalyScore(
                invoice_id=record.invoice_id,
                amount=record.amount,
                z_score=z_score,
                is_anomaly=abs(z_score) > threshold,
            )
        )
    return scores
