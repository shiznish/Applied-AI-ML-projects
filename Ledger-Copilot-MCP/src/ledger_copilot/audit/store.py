from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime

from ledger_copilot.audit.schema import AuditRecord

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS audit_log (
    record_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    actor_user_id TEXT,
    on_behalf_of_chain TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    arguments TEXT NOT NULL,
    scopes_presented TEXT NOT NULL,
    decision TEXT NOT NULL,
    approval_id TEXT,
    idempotency_key TEXT,
    latency_ms REAL NOT NULL,
    result_status TEXT NOT NULL
)
"""

# SQLite has no built-in column/row immutability, so "append-only" is enforced
# with triggers that abort any UPDATE/DELETE against the table rather than
# relying on every caller only ever using `append()`. In production (Postgres
# + S3 Object Lock) the equivalent guarantee is a Postgres REVOKE of
# UPDATE/DELETE on this table from the application role, plus WORM retention
# on the copy of the log archived to S3 under Object Lock — this module is
# the local/dev stand-in for that pair.
_CREATE_NO_UPDATE_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS audit_log_no_update
BEFORE UPDATE ON audit_log
BEGIN
    SELECT RAISE(ABORT, 'audit_log is append-only: UPDATE is not permitted');
END
"""

_CREATE_NO_DELETE_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS audit_log_no_delete
BEFORE DELETE ON audit_log
BEGIN
    SELECT RAISE(ABORT, 'audit_log is append-only: DELETE is not permitted');
END
"""

_INSERT = """
INSERT INTO audit_log (
    record_id, timestamp, trace_id, actor_user_id, on_behalf_of_chain,
    tool_name, arguments, scopes_presented, decision, approval_id,
    idempotency_key, latency_ms, result_status
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_SELECT_ALL = """
SELECT
    record_id, timestamp, trace_id, actor_user_id, on_behalf_of_chain,
    tool_name, arguments, scopes_presented, decision, approval_id,
    idempotency_key, latency_ms, result_status
FROM audit_log
"""


def _row_to_record(row: tuple[object, ...]) -> AuditRecord:
    (
        record_id,
        timestamp,
        trace_id,
        actor_user_id,
        on_behalf_of_chain,
        tool_name,
        arguments,
        scopes_presented,
        decision,
        approval_id,
        idempotency_key,
        latency_ms,
        result_status,
    ) = row
    return AuditRecord(
        record_id=str(record_id),
        timestamp=datetime.fromisoformat(str(timestamp)),
        trace_id=str(trace_id),
        actor_user_id=None if actor_user_id is None else str(actor_user_id),
        on_behalf_of_chain=json.loads(str(on_behalf_of_chain)),
        tool_name=str(tool_name),
        arguments=json.loads(str(arguments)),
        scopes_presented=json.loads(str(scopes_presented)),
        decision=str(decision),  # type: ignore[arg-type]
        approval_id=None if approval_id is None else str(approval_id),
        idempotency_key=None if idempotency_key is None else str(idempotency_key),
        latency_ms=float(latency_ms),  # type: ignore[arg-type]
        result_status=str(result_status),  # type: ignore[arg-type]
    )


class AuditStore:
    """Append-only SQLite-backed audit log.

    `database_path=":memory:"` (the default) is appropriate for tests and
    for local/dev runs that don't need audit records to survive a restart;
    pass a file path to persist them. Swapping to Postgres + S3 Object Lock
    for production is future work — this class's `append`/`query` interface
    is the seam that swap would happen behind.
    """

    def __init__(self, database_path: str = ":memory:") -> None:
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(database_path, check_same_thread=False)
        with self._lock:
            self._connection.execute(_CREATE_TABLE)
            self._connection.execute(_CREATE_NO_UPDATE_TRIGGER)
            self._connection.execute(_CREATE_NO_DELETE_TRIGGER)
            self._connection.commit()

    def append(self, record: AuditRecord) -> None:
        with self._lock:
            self._connection.execute(
                _INSERT,
                (
                    record.record_id,
                    record.timestamp.isoformat(),
                    record.trace_id,
                    record.actor_user_id,
                    json.dumps(record.on_behalf_of_chain),
                    record.tool_name,
                    json.dumps(record.arguments),
                    json.dumps(record.scopes_presented),
                    record.decision,
                    record.approval_id,
                    record.idempotency_key,
                    record.latency_ms,
                    record.result_status,
                ),
            )
            self._connection.commit()

    def get(self, record_id: str) -> AuditRecord | None:
        with self._lock:
            cursor = self._connection.execute(
                f"{_SELECT_ALL} WHERE record_id = ?", (record_id,)
            )
            row = cursor.fetchone()
        return None if row is None else _row_to_record(row)

    def ping(self) -> bool:
        """Cheap liveness check for the readiness probe: True if a trivial
        query against the connection still succeeds, False on any sqlite3
        error (e.g. the connection was forced closed to simulate an outage)."""
        try:
            with self._lock:
                self._connection.execute("SELECT 1")
            return True
        except sqlite3.Error:
            return False

    def query(self, tool_name: str | None = None) -> list[AuditRecord]:
        with self._lock:
            if tool_name is None:
                cursor = self._connection.execute(_SELECT_ALL)
            else:
                cursor = self._connection.execute(
                    f"{_SELECT_ALL} WHERE tool_name = ?", (tool_name,)
                )
            rows = cursor.fetchall()
        return [_row_to_record(row) for row in rows]
