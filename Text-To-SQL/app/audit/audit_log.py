"""Immutable per-query audit trail.

Why JSONL, not a Postgres table: every other durable, append-mostly record in
this project (logs/queries.jsonl, logs/fewshot.json) already lives as a
JSON(-lines) file under settings.log_dir rather than in Postgres, and this
sandbox has no live database to migrate against or test against anyway. A
JSONL file also gives "immutable" almost for free — appends only, no UPDATE
statement anyone could (mis)use to rewrite history. The tradeoffs a real
production deployment would want back (indexed queries by user/date range,
retention/rotation, concurrent-writer safety beyond POSIX append semantics)
are exactly why storage is behind the `AuditStorage` interface below: a
`PostgresAuditStorage` implementing the same `append`/`read_all` shape is a
drop-in swap later (e.g. once audit rows need to be joined against traces for
deeper observability) without touching call sites in app/main.py or the
GET /v1/admin/audit route.

`AuditRecord.result_hash` intentionally hashes a canonical JSON serialization
of columns+rows (sha256) rather than storing rows verbatim — the audit trail
should prove "this exact result set was returned" (e.g. for later dispute
resolution or drift detection) without duplicating potentially sensitive
result data a second time outside the RLS-governed query path.
"""
import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Protocol

from ..config import settings


@dataclass(frozen=True)
class AuditRecord:
    correlation_id: str
    query_id: str
    timestamp: float
    username: str
    role: str
    region: str
    question: str | None = None
    status: str = "ok"                       # ok | error | blocked | ambiguous | complexity_rejected
    generated_sql: str | None = None          # what the LLM first produced (None for /v1/execute)
    executed_sql: str | None = None           # what actually ran, post-guardrail/post-repair
    repair_attempts: int = 0
    row_count: int | None = None
    result_hash: str | None = None
    confidence: float | None = None           # None for /v1/execute (no LLM/confidence involved)
    confidence_breakdown: dict | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def result_hash(columns: list[str], rows: list[list]) -> str:
    """Deterministic sha256 over a canonical JSON serialization of columns+rows.
    Same columns/rows (any row order fed in, since callers pass the executed
    result's own order) always hash identically; any difference in either
    changes the hash. `default=str` covers non-JSON-native types the DB driver
    can return (datetime, Decimal, etc.) without the hash depending on which
    of those subclasses showed up."""
    canonical = json.dumps({"columns": columns, "rows": rows}, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class AuditStorage(Protocol):
    """Swappable storage backend — see module docstring for why JSONL is the
    default and how a Postgres-backed implementation would slot in later."""

    def append(self, record: dict) -> None: ...

    def read_all(self, limit: int = 50) -> list[dict]: ...


@dataclass
class JsonlAuditStorage:
    """Append-only JSON-lines file, matching this project's existing
    logs/*.jsonl convention (see app/main.py's _QUERY_LOG)."""

    path: str

    def append(self, record: dict) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def read_all(self, limit: int = 50) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        records: list[dict] = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        # Newest-first, matching GET /v1/history's convention.
        return list(reversed(records[-limit:])) if limit else list(reversed(records))


@dataclass
class AuditLog:
    """Thin façade so app/main.py and tests depend on this, not on a specific
    storage backend."""

    storage: AuditStorage = field(default_factory=lambda: JsonlAuditStorage(_default_path()))

    def write(self, record: AuditRecord) -> None:
        self.storage.append(record.to_dict())

    def read_all(self, limit: int = 50) -> list[dict]:
        return self.storage.read_all(limit)


def _default_path() -> str:
    return os.path.join(settings.log_dir, "audit.jsonl")


default_log = AuditLog(JsonlAuditStorage(_default_path()))


def write(record: AuditRecord, log: AuditLog | None = None) -> None:
    (log or default_log).write(record)


def read_all(limit: int = 50, log: AuditLog | None = None) -> list[dict]:
    return (log or default_log).read_all(limit)
