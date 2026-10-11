from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

DEFAULT_FIXTURE_PATH = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "purchase_orders.json"
)
DEFAULT_INVOICE_FIXTURE_PATH = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "invoices.json"
)
DEFAULT_BUDGET_FIXTURE_PATH = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "budgets.json"
)

_SCHEMA = """
CREATE TABLE purchase_orders (
    po_id TEXT PRIMARY KEY,
    vendor_name TEXT NOT NULL,
    status TEXT NOT NULL,
    order_date TEXT NOT NULL,
    currency TEXT NOT NULL,
    cost_center TEXT
);

CREATE TABLE purchase_order_line_items (
    po_id TEXT NOT NULL REFERENCES purchase_orders(po_id),
    line_number INTEGER NOT NULL,
    description TEXT NOT NULL,
    quantity_ordered REAL NOT NULL,
    quantity_received REAL NOT NULL,
    unit_price REAL NOT NULL,
    PRIMARY KEY (po_id, line_number)
);

CREATE TABLE invoices (
    invoice_id TEXT PRIMARY KEY,
    po_id TEXT NOT NULL,
    vendor_name TEXT NOT NULL,
    created_by TEXT NOT NULL,
    amount REAL NOT NULL,
    currency TEXT NOT NULL,
    match_discrepancy INTEGER NOT NULL,
    invoice_date TEXT,
    cost_center TEXT
);

CREATE TABLE budgets (
    cost_center TEXT NOT NULL,
    period_start TEXT NOT NULL,
    period_end TEXT NOT NULL,
    budgeted_amount REAL NOT NULL
);
"""


class ErpClient:
    """Fixture-seeded ERP client backed by a real SQLite database.

    The JSON fixture files stay the human-readable, git-diffable source of
    seed data, but every lookup below is a genuine parameterized SQL query
    against an in-memory `sqlite3` connection loaded once at construction —
    not a JSON re-parse per call."""

    def __init__(
        self,
        fixture_path: Path = DEFAULT_FIXTURE_PATH,
        invoice_fixture_path: Path = DEFAULT_INVOICE_FIXTURE_PATH,
        budget_fixture_path: Path = DEFAULT_BUDGET_FIXTURE_PATH,
    ) -> None:
        # check_same_thread=False: FastMCP runs sync tool functions on a
        # worker thread different from the one that constructs this module-
        # level client. Safe here since each instance's data is seeded once
        # at construction and never written to again.
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._seed_purchase_orders(json.loads(fixture_path.read_text()))
        self._seed_invoices(json.loads(invoice_fixture_path.read_text()))
        self._seed_budgets(json.loads(budget_fixture_path.read_text()))
        self._conn.commit()

    def _seed_purchase_orders(self, records: list[dict[str, Any]]) -> None:
        for record in records:
            self._conn.execute(
                "INSERT INTO purchase_orders "
                "(po_id, vendor_name, status, order_date, currency, cost_center) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    record["po_id"],
                    record["vendor_name"],
                    record["status"],
                    record["order_date"],
                    record["currency"],
                    record.get("cost_center"),
                ),
            )
            for item in record["line_items"]:
                self._conn.execute(
                    "INSERT INTO purchase_order_line_items "
                    "(po_id, line_number, description, quantity_ordered, "
                    "quantity_received, unit_price) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        record["po_id"],
                        item["line_number"],
                        item["description"],
                        item["quantity_ordered"],
                        item["quantity_received"],
                        item["unit_price"],
                    ),
                )

    def _seed_invoices(self, records: list[dict[str, Any]]) -> None:
        for record in records:
            self._conn.execute(
                "INSERT INTO invoices "
                "(invoice_id, po_id, vendor_name, created_by, amount, currency, "
                "match_discrepancy, invoice_date, cost_center) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record["invoice_id"],
                    record["po_id"],
                    record["vendor_name"],
                    record["created_by"],
                    record["amount"],
                    record["currency"],
                    int(record["match_discrepancy"]),
                    record.get("invoice_date"),
                    record.get("cost_center"),
                ),
            )

    def _seed_budgets(self, records: list[dict[str, Any]]) -> None:
        for record in records:
            self._conn.execute(
                "INSERT INTO budgets "
                "(cost_center, period_start, period_end, budgeted_amount) "
                "VALUES (?, ?, ?, ?)",
                (
                    record["cost_center"],
                    record["period_start"],
                    record["period_end"],
                    record["budgeted_amount"],
                ),
            )

    def _line_items_for(self, po_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT line_number, description, quantity_ordered, quantity_received, "
            "unit_price FROM purchase_order_line_items WHERE po_id = ? "
            "ORDER BY line_number",
            (po_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_purchase_order(self, po_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT po_id, vendor_name, status, order_date, currency, cost_center "
            "FROM purchase_orders WHERE po_id = ?",
            (po_id,),
        ).fetchone()
        if row is None:
            return None
        record = dict(row)
        record["line_items"] = self._line_items_for(po_id)
        return record

    def list_purchase_orders(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT po_id, vendor_name, status, order_date, currency, cost_center "
            "FROM purchase_orders"
        ).fetchall()
        records = []
        for row in rows:
            record = dict(row)
            record["line_items"] = self._line_items_for(record["po_id"])
            records.append(record)
        return records

    def list_invoices(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT invoice_id, po_id, vendor_name, created_by, amount, currency, "
            "match_discrepancy, invoice_date, cost_center FROM invoices"
        ).fetchall()
        records = []
        for row in rows:
            record = dict(row)
            record["match_discrepancy"] = bool(record["match_discrepancy"])
            records.append(record)
        return records

    def list_budgets(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT cost_center, period_start, period_end, budgeted_amount FROM budgets"
        ).fetchall()
        return [dict(row) for row in rows]
